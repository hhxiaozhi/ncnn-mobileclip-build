// siglip2_infer.c v4 —— 固定 256 字节拷贝 tensor，手动按 2.51 布局写 clientBuf
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
    if (!h) { fprintf(stderr, "[pd] dlopen: %s\n", dlerror()); return -1; }
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

#define TENSOR_SIZE 256
#define OFF_ID      8
#define OFF_NAME    16
#define OFF_TYPE    24
#define OFF_DTYPE   32
#define OFF_RANK    80
#define OFF_DIMS    88
#define OFF_BUF     104
#define OFF_BUFSZ   112

static void dump_tensor_raw(const char* tag, const uint8_t* p) {
    uint32_t ver = *(uint32_t*)(p + 0);
    uint32_t id  = *(uint32_t*)(p + OFF_ID);
    uint64_t np  = *(uint64_t*)(p + OFF_NAME);
    uint32_t ty  = *(uint32_t*)(p + OFF_TYPE);
    uint32_t dt  = *(uint32_t*)(p + OFF_DTYPE);
    uint32_t rk  = *(uint32_t*)(p + OFF_RANK);
    uint64_t dp  = *(uint64_t*)(p + OFF_DIMS);
    fprintf(stderr, "[%s] v=%u id=%u name=%s type=%u dtype=0x%x rank=%u dims=[",
            tag, ver, id, np ? (const char*)np : "?", ty, dt, rk);
    if (dp && rk <= 8) {
        uint32_t* d = (uint32_t*)dp;
        for (uint32_t i = 0; i < rk; i++) fprintf(stderr, "%u%s", d[i], i+1<rk?",":"");
    }
    fprintf(stderr, "]\n");
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <bin> <in> [out] [iters]\n", argv[0]); return 1; }
    const char* bin_path = argv[1];
    const char* in_path  = argv[2];
    const char* out_path = argc > 3 ? argv[3] : "/tmp/siglip2_out.bin";
    int iters = argc > 4 ? atoi(argv[4]) : 1;

    fprintf(stderr, "[info] sizeof(Qnn_Tensor_t)=%zu  (runtime expects 256)\n", sizeof(Qnn_Tensor_t));

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
    fprintf(stderr, "[qnn] %s\n", pv[0]->providerName ? pv[0]->providerName : "?");

    typedef Qnn_ErrorHandle_t (*sys_gp_t)(const QnnSystemInterface_t***, uint32_t*);
    sys_gp_t sys_gp = (sys_gp_t)dlsym(h_sys, "QnnSystemInterface_getProviders");
    const QnnSystemInterface_t** spv = NULL; uint32_t snum = 0;
    e = sys_gp(&spv, &snum);
    if (e != QNN_SUCCESS || snum == 0) return 2;
    QNN_SYSTEM_INTERFACE_VER_TYPE sys = spv[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

    Qnn_LogHandle_t log = NULL;
    qnn.logCreate(NULL, QNN_LOG_LEVEL_ERROR, &log);
    Qnn_BackendHandle_t be = NULL;
    e = qnn.backendCreate(log, NULL, &be);
    if (e != QNN_SUCCESS) { fprintf(stderr, "[be] %lu\n", (unsigned long)e); return 3; }
    Qnn_DeviceHandle_t dev = NULL;
    e = qnn.deviceCreate(log, NULL, &dev);
    if (e != QNN_SUCCESS) { fprintf(stderr, "[dev] %lu\n", (unsigned long)e); return 3; }

    size_t bin_sz = 0;
    void* bin_buf = read_file(bin_path, &bin_sz);
    if (!bin_buf) return 4;

    QnnSystemContext_Handle_t sysctx = NULL;
    e = sys.systemContextCreate(&sysctx);
    const QnnSystemContext_BinaryInfo_t* bi = NULL;
    Qnn_ContextBinarySize_t bi_sz = 0;
    e = sys.systemContextGetBinaryInfo(sysctx, bin_buf, bin_sz, &bi, &bi_sz);
    if (e != QNN_SUCCESS || !bi) { fprintf(stderr, "[binInfo] %lu\n", (unsigned long)e); return 5; }

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
    if (!g3) { fprintf(stderr, "[x] graph not found\n"); return 6; }

    Qnn_ContextHandle_t ctx = NULL;
    e = qnn.contextCreateFromBinary(be, dev, NULL, bin_buf, bin_sz, &ctx, NULL);
    if (e != QNN_SUCCESS) { fprintf(stderr, "[ctx] %lu\n", (unsigned long)e); return 6; }
    Qnn_GraphHandle_t graph = NULL;
    e = qnn.graphRetrieve(ctx, "siglip2_model", &graph);
    if (e != QNN_SUCCESS) { fprintf(stderr, "[graph] %lu\n", (unsigned long)e); return 6; }

    // ---- 固定 256 字节拷贝真实 tensor ----
    uint8_t in_raw[TENSOR_SIZE] = {0};
    uint8_t out_raw[TENSOR_SIZE] = {0};
    memcpy(in_raw,  &g3->graphInputs[0],  TENSOR_SIZE);
    memcpy(out_raw, &g3->graphOutputs[0], TENSOR_SIZE);

    fprintf(stderr, "[tensor] sizeof(Qnn_Tensor_t)=%zu TENSOR_SIZE=256\n", sizeof(Qnn_Tensor_t));
    dump_tensor_raw("in_real",  in_raw);
    dump_tensor_raw("out_real", out_raw);

    size_t in_read = 0;
    void* in_data = read_file(in_path, &in_read);
    if (!in_data) return 7;

    // 推算字节数（fp16）
    uint32_t in_elems = 1;
    uint32_t* idims = *(uint32_t**)(in_raw + OFF_DIMS);
    uint32_t irank = *(uint32_t*)(in_raw + OFF_RANK);
    for (uint32_t i = 0; i < irank; i++) in_elems *= idims[i];
    uint32_t in_bytes = in_elems * 2;

    uint32_t out_elems = 1;
    uint32_t* odims = *(uint32_t**)(out_raw + OFF_DIMS);
    uint32_t orank = *(uint32_t*)(out_raw + OFF_RANK);
    for (uint32_t i = 0; i < orank; i++) out_elems *= odims[i];
    uint32_t out_bytes = out_elems * 2;

    fprintf(stderr, "[in] want=%u got=%zu\n", in_bytes, in_read);
    fprintf(stderr, "[out] bytes=%u\n", out_bytes);

    void* out_data = calloc(1, out_bytes);
    if (!out_data) return 7;

    // 手动写 clientBuf（不通过 struct 访问，避开 2.31/2.51 布局差异）
    *(uint64_t*)(in_raw  + OFF_BUF) = (uint64_t)in_data;
    *(uint32_t*)(in_raw  + OFF_BUFSZ) = in_bytes;
    *(uint64_t*)(out_raw + OFF_BUF) = (uint64_t)out_data;
    *(uint32_t*)(out_raw + OFF_BUFSZ) = out_bytes;

    double a = now_ms();
    e = qnn.graphExecute(graph, (const Qnn_Tensor_t*)in_raw, 1, (const Qnn_Tensor_t*)out_raw, 1, NULL, NULL);
    double b = now_ms();
    fprintf(stderr, "[exec #1] err=%lu %.2f ms\n", (unsigned long)e, b - a);
    if (e != QNN_SUCCESS) {
        // 打印 in_raw 前 256 字节 hex 便于排查
        fprintf(stderr, "[in_raw] hex: ");
        for (int i = 0; i < 128; i++) fprintf(stderr, "%02x", in_raw[i]);
        fprintf(stderr, "\n");
        return 9;
    }

    double total = 0;
    for (int i = 0; i < iters; i++) {
        a = now_ms();
        e = qnn.graphExecute(graph, (const Qnn_Tensor_t*)in_raw, 1, (const Qnn_Tensor_t*)out_raw, 1, NULL, NULL);
        b = now_ms();
        if (e != QNN_SUCCESS) break;
        total += (b - a);
    }
    if (iters > 0) fprintf(stderr, "[exec avg %d] %.2f ms\n", iters, total / iters);

    FILE* of = fopen(out_path, "wb");
    if (of) { fwrite(out_data, 1, out_bytes, of); fclose(of); fprintf(stderr, "[out] %s %u bytes\n", out_path, out_bytes); }

    uint16_t* o16 = (uint16_t*)out_data;
    fprintf(stderr, "[out] first 8 fp16: ");
    for (uint32_t i = 0; i < 8 && i*2 < out_bytes; i++) fprintf(stderr, "0x%04x ", o16[i]);
    fprintf(stderr, "\n");

    free(in_data); free(out_data); free(bin_buf);
    return 0;
}
