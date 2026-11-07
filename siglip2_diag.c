// siglip2_diag.c — 只读诊断：确定 graphOutputs 数组真实 stride + dump 完整 tensor 内存
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>

#include "QnnInterface.h"
#include "QnnTypes.h"
#include "QnnSystemInterface.h"
#include "QnnSystemContext.h"

struct remote_rpc_control_unsigned_module { int32_t domain; int32_t enable; };
#define DSPRPC_CONTROL_UNSIGNED_MODULE 2

static int enable_unsigned_pd(void) {
    void* h = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) return -1;
    typedef int (*rsc_t)(uint32_t, void*, uint32_t);
    rsc_t rsc = (rsc_t)dlsym(h, "remote_session_control");
    if (!rsc) return -1;
    struct remote_rpc_control_unsigned_module um = {3, 1};
    return rsc(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof(um));
}

static void* read_file(const char* path, size_t* sz) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc(n);
    if (!p) { fclose(f); return NULL; }
    if (fread(p, 1, n, f) != (size_t)n) { free(p); fclose(f); return NULL; }
    fclose(f); *sz = (size_t)n; return p;
}

static void hexdump(const char* tag, const uint8_t* p, int n) {
    fprintf(stderr, "[%s] (%d bytes)\n", tag, n);
    for (int i = 0; i < n; i += 16) {
        fprintf(stderr, "  %04d: ", i);
        for (int j = 0; j < 16 && i + j < n; j++) fprintf(stderr, "%02x ", p[i+j]);
        fprintf(stderr, "\n");
    }
}

// 在 buf+off 处按 Qnn_Tensor_t 解析，打印关键字段
static void probe(const char* tag, const uint8_t* buf, int off) {
    const uint8_t* p = buf + off;
    uint32_t ver = *(uint32_t*)(p + 0);
    uint32_t id  = *(uint32_t*)(p + 8);
    char*    nm  = *(char**)(p + 16);
    uint32_t ty  = *(uint32_t*)(p + 24);
    uint32_t dt  = *(uint32_t*)(p + 32);
    uint32_t rk  = *(uint32_t*)(p + 80);
    fprintf(stderr, "  [probe @%d] ver=%u id=%u type=%u dtype=0x%x rank=%u name=%s\n",
            off, ver, id, ty, dt, rk, (nm && (uintptr_t)nm > 0x1000) ? nm : "?");
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <bin>\n", argv[0]); return 1; }
    const char* bin_path = argv[1];

    setenv("ADSP_LIBRARY_PATH", "/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp", 1);
    fprintf(stderr, "[pd] r=%d\n", enable_unsigned_pd());

    const char* qdir = getenv("QNN_LIB_DIR");
    if (!qdir) qdir = "/data/data/com.termux/files/home/qnn_v51";
    char s_sys[512], s_stub[512], s_htp[512];
    snprintf(s_sys, sizeof(s_sys), "%s/libQnnSystem.so", qdir);
    snprintf(s_stub, sizeof(s_stub), "%s/libQnnHtpV68Stub.so", qdir);
    snprintf(s_htp, sizeof(s_htp), "%s/libQnnHtp.so", qdir);
    void* h_sys = dlopen(s_sys, RTLD_NOW|RTLD_GLOBAL);
    if (!h_sys) { fprintf(stderr, "[qnn] sys: %s\n", dlerror()); return 2; }
    if (!dlopen(s_stub, RTLD_NOW|RTLD_GLOBAL)) { fprintf(stderr, "[qnn] stub: %s\n", dlerror()); return 2; }
    void* h_htp = dlopen(s_htp, RTLD_NOW|RTLD_GLOBAL);
    if (!h_htp) { fprintf(stderr, "[qnn] htp: %s\n", dlerror()); return 2; }

    typedef Qnn_ErrorHandle_t (*gp_t)(const QnnInterface_t***, uint32_t*);
    gp_t gp = (gp_t)dlsym(h_htp, "QnnInterface_getProviders");
    const QnnInterface_t** pv = NULL; uint32_t num = 0;
    if (gp(&pv, &num) != QNN_SUCCESS || num == 0) return 2;
    QNN_INTERFACE_VER_TYPE qnn = pv[0]->QNN_INTERFACE_VER_NAME;

    typedef Qnn_ErrorHandle_t (*sys_gp_t)(const QnnSystemInterface_t***, uint32_t*);
    sys_gp_t sys_gp = (sys_gp_t)dlsym(h_sys, "QnnSystemInterface_getProviders");
    const QnnSystemInterface_t** spv = NULL; uint32_t snum = 0;
    sys_gp(&spv, &snum);
    QNN_SYSTEM_INTERFACE_VER_TYPE sys = spv[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

    Qnn_LogHandle_t log = NULL;
    qnn.logCreate(NULL, QNN_LOG_LEVEL_ERROR, &log);
    Qnn_BackendHandle_t be = NULL;
    if (qnn.backendCreate(log, NULL, &be) != QNN_SUCCESS) return 3;
    Qnn_DeviceHandle_t dev = NULL;
    if (qnn.deviceCreate(log, NULL, &dev) != QNN_SUCCESS) return 3;

    size_t bin_sz = 0;
    void* bin_buf = read_file(bin_path, &bin_sz);
    if (!bin_buf) return 4;

    QnnSystemContext_Handle_t sysctx = NULL;
    sys.systemContextCreate(&sysctx);
    const QnnSystemContext_BinaryInfo_t* bi = NULL;
    Qnn_ContextBinarySize_t bi_sz = 0;
    if (sys.systemContextGetBinaryInfo(sysctx, bin_buf, bin_sz, &bi, &bi_sz) != QNN_SUCCESS || !bi) return 5;

    uint32_t ng = 0; const QnnSystemContext_GraphInfo_t* graphs = NULL;
    if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        ng = bi->contextBinaryInfoV3.numGraphs; graphs = bi->contextBinaryInfoV3.graphs;
    } else if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        ng = bi->contextBinaryInfoV2.numGraphs; graphs = bi->contextBinaryInfoV2.graphs;
    } else {
        ng = bi->contextBinaryInfoV1.numGraphs; graphs = bi->contextBinaryInfoV1.graphs;
    }

    const QnnSystemContext_GraphInfoV3_t* g3 = NULL;
    for (uint32_t i = 0; i < ng && graphs; i++) {
        if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3 &&
            graphs[i].graphInfoV3.graphName &&
            strcmp(graphs[i].graphInfoV3.graphName, "siglip2_model") == 0) {
            g3 = &graphs[i].graphInfoV3; break;
        }
    }
    if (!g3) return 6;

    fprintf(stderr, "\n=== sizeof(Qnn_Tensor_t) = %zu ===\n", sizeof(Qnn_Tensor_t));
    fprintf(stderr, "graphName=%s in=%u out=%u\n",
            g3->graphName, g3->numGraphInputs, g3->numGraphOutputs);
    fprintf(stderr, "graphInputs ptr=%p  graphOutputs ptr=%p\n",
            (void*)g3->graphInputs, (void*)g3->graphOutputs);

    // 1) dump graphInputs[0] 起始 384 字节
    fprintf(stderr, "\n===== graphInputs DUMP (384 bytes from array start) =====\n");
    hexdump("inputs_raw", (const uint8_t*)g3->graphInputs, 384);

    // 2) dump graphOutputs 起始 768 字节
    fprintf(stderr, "\n===== graphOutputs DUMP (768 bytes from array start) =====\n");
    hexdump("outputs_raw", (const uint8_t*)g3->graphOutputs, 768);

    // 3) 逐个 offset 试探，确定 stride
    fprintf(stderr, "\n===== stride probe on graphOutputs =====\n");
    int cand[] = {0, 144, 256, 288, 512, 120, 128, 160, 192, 200, 240};
    for (unsigned k = 0; k < sizeof(cand)/sizeof(cand[0]); k++) probe("out", (const uint8_t*)g3->graphOutputs, cand[k]);

    fprintf(stderr, "\n===== stride probe on graphInputs =====\n");
    for (unsigned k = 0; k < sizeof(cand)/sizeof(cand[0]); k++) probe("in", (const uint8_t*)g3->graphInputs, cand[k]);

    free(bin_buf);
    fprintf(stderr, "\n[diag done]\n");
    return 0;
}
