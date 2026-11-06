// siglip2_infer.c v8 —— 完全照搬 MobileCLIP S2 成功方案
// 关键：TCOPY=256（QNN 2.51 runtime tensor 大小，不是 2.31 头文件的 144）
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <time.h>

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
    int r = rsc(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof(um));
    fprintf(stderr, "[pd] r=%d\n", r);
    return r;
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

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

#define TCOPY 256

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <bin> <in> <out_prefix> [iters]\n", argv[0]); return 1; }
    const char* bin_path = argv[1];
    const char* in_path  = argv[2];
    const char* out_prefix = argv[3];
    int iters = argc > 4 ? atoi(argv[4]) : 1;

    setenv("ADSP_LIBRARY_PATH", "/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp", 1);
    enable_unsigned_pd();

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
    Qnn_ErrorHandle_t e = gp(&pv, &num);
    if (e != QNN_SUCCESS || num == 0) return 2;
    QNN_INTERFACE_VER_TYPE qnn = pv[0]->QNN_INTERFACE_VER_NAME;

    typedef Qnn_ErrorHandle_t (*sys_gp_t)(const QnnSystemInterface_t***, uint32_t*);
    sys_gp_t sys_gp = (sys_gp_t)dlsym(h_sys, "QnnSystemInterface_getProviders");
    const QnnSystemInterface_t** spv = NULL; uint32_t snum = 0;
    sys_gp(&spv, &snum);
    QNN_SYSTEM_INTERFACE_VER_TYPE sys = spv[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

    Qnn_LogHandle_t log = NULL;
    qnn.logCreate(NULL, QNN_LOG_LEVEL_ERROR, &log);
    Qnn_BackendHandle_t be = NULL;
    e = qnn.backendCreate(log, NULL, &be);
    if (e != QNN_SUCCESS) return 3;
    Qnn_DeviceHandle_t dev = NULL;
    e = qnn.deviceCreate(log, NULL, &dev);
    if (e != QNN_SUCCESS) return 3;

    size_t bin_sz = 0;
    void* bin_buf = read_file(bin_path, &bin_sz);
    if (!bin_buf) return 4;

    QnnSystemContext_Handle_t sysctx = NULL;
    sys.systemContextCreate(&sysctx);
    const QnnSystemContext_BinaryInfo_t* bi = NULL;
    Qnn_ContextBinarySize_t bi_sz = 0;
    e = sys.systemContextGetBinaryInfo(sysctx, bin_buf, bin_sz, &bi, &bi_sz);
    if (e != QNN_SUCCESS || !bi) return 5;

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
        if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
            if (graphs[i].graphInfoV3.graphName && strcmp(graphs[i].graphInfoV3.graphName, "siglip2_model") == 0) {
                g3 = &graphs[i].graphInfoV3; break;
            }
        }
    }
    if (!g3) return 6;

    fprintf(stderr, "[graph] in=%u out=%u TCOPY=%d sizeof(Qnn_Tensor_t)=%zu\n",
            g3->numGraphInputs, g3->numGraphOutputs, TCOPY, sizeof(Qnn_Tensor_t));

    Qnn_Tensor_t* in_231 = &g3->graphInputs[0];
    uint32_t in_elems = 1;
    for (uint32_t i = 0; i < in_231->v2.rank; i++) in_elems *= in_231->v2.dimensions[i];
    uint32_t in_bytes = in_elems * 2;
    fprintf(stderr, "[in] id=%u name=%s rank=%u bytes=%u\n",
            in_231->v2.id, in_231->v2.name ? in_231->v2.name : "?", in_231->v2.rank, in_bytes);

    uint32_t nout = g3->numGraphOutputs;
    if (nout > 8) nout = 8;
    uint32_t out_bytes_arr[8] = {0};
    void* out_data_arr[8] = {NULL};
    uint8_t out_raw_arr[8][TCOPY];

    for (uint32_t i = 0; i < nout; i++) {
        memset(out_raw_arr[i], 0, TCOPY);
        Qnn_Tensor_t* o = &g3->graphOutputs[i];
        uint32_t ne = 1;
        for (uint32_t j = 0; j < o->v2.rank; j++) ne *= o->v2.dimensions[j];
        out_bytes_arr[i] = ne * 2;
        out_data_arr[i] = calloc(1, out_bytes_arr[i]);
        memcpy(out_raw_arr[i], o, sizeof(Qnn_Tensor_t));  // 先拷真实 144 字节
        // 剩下 112 字节保持 0
        *(uint64_t*)(out_raw_arr[i] + 104) = (uint64_t)out_data_arr[i];
        *(uint32_t*)(out_raw_arr[i] + 112) = out_bytes_arr[i];
        fprintf(stderr, "[out%d] id=%u name=%s rank=%u bytes=%u\n",
                i, o->v2.id, o->v2.name ? o->v2.name : "?", o->v2.rank, out_bytes_arr[i]);
    }

    Qnn_ContextHandle_t ctx = NULL;
    e = qnn.contextCreateFromBinary(be, dev, NULL, bin_buf, bin_sz, &ctx, NULL);
    if (e != QNN_SUCCESS) return 7;
    Qnn_GraphHandle_t graph = NULL;
    e = qnn.graphRetrieve(ctx, "siglip2_model", &graph);
    if (e != QNN_SUCCESS) return 7;

    size_t in_read = 0;
    void* in_data = read_file(in_path, &in_read);
    if (!in_data || in_read < in_bytes) return 8;

    uint8_t in_raw[TCOPY];
    memset(in_raw, 0, TCOPY);
    memcpy(in_raw, in_231, sizeof(Qnn_Tensor_t));  // 拷真实 144 字节，剩 112 零
    *(uint64_t*)(in_raw + 104) = (uint64_t)in_data;
    *(uint32_t*)(in_raw + 112) = in_bytes;

    // 输出连续排（每个 256 字节）
    uint8_t* out_pair = calloc(1, TCOPY * nout);
    for (uint32_t i = 0; i < nout; i++) {
        memcpy(out_pair + TCOPY * i, out_raw_arr[i], TCOPY);
    }

    typedef Qnn_ErrorHandle_t (*gexec_t)(Qnn_GraphHandle_t, const void*, uint32_t, void*, uint32_t, void*, void*);
    gexec_t gexec = (gexec_t)qnn.graphExecute;

    double a = now_ms();
    e = gexec(graph, in_raw, 1, out_pair, nout, NULL, NULL);
    double b = now_ms();
    fprintf(stderr, "[exec #1] err=%lu %.2f ms\n", (unsigned long)e, b - a);
    if (e != QNN_SUCCESS) return 9;

    double total = 0; int ok = 0;
    for (int i = 0; i < iters; i++) {
        a = now_ms();
        e = gexec(graph, in_raw, 1, out_pair, nout, NULL, NULL);
        b = now_ms();
        if (e != QNN_SUCCESS) break;
        total += (b - a); ok++;
    }
    if (ok > 0) fprintf(stderr, "[exec avg %d] %.2f ms\n", ok, total / ok);

    for (uint32_t i = 0; i < nout; i++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%u.bin", out_prefix, i);
        FILE* of = fopen(path, "wb");
        if (of) { fwrite(out_data_arr[i], 1, out_bytes_arr[i], of); fclose(of); }
        fprintf(stderr, "[saved] %s (%u bytes)\n", path, out_bytes_arr[i]);
    }

    if (nout >= 2) {
        uint16_t* o16 = (uint16_t*)out_data_arr[1];
        fprintf(stderr, "[pooler] first 8: ");
        for (int i = 0; i < 8 && (uint32_t)(i*2) < out_bytes_arr[1]; i++)
            fprintf(stderr, "0x%04x ", o16[i]);
        fprintf(stderr, "\n");
    }

    free(in_data);
    for (uint32_t i = 0; i < nout; i++) free(out_data_arr[i]);
    free(out_pair); free(bin_buf);
    return 0;
}
