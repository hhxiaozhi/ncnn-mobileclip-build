// siglip2_infer.c v3 —— 适配 QNN 2.31 QnnSystemContext API
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
    if (!rsc) { fprintf(stderr, "[pd] no rsc\n"); return -1; }
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

static void dump_tensor(const char* tag, const Qnn_Tensor_t* t) {
    if (!t) return;
    const Qnn_TensorV2_t* v = &t->v2;
    fprintf(stderr, "[%s] v=%u id=%u name=%s type=%d dtype=0x%x rank=%u dims=[",
            tag, t->version, v->id, v->name ? v->name : "?", (int)v->type,
            (unsigned)v->dataType, v->rank);
    for (uint32_t i = 0; i < v->rank; i++) fprintf(stderr, "%u%s", v->dimensions[i], i+1<v->rank?",":"");
    fprintf(stderr, "]\n");
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <ctx_bin> <input_fp16_nhwc> [out] [iters]\n", argv[0]);
        return 1;
    }
    const char* bin_path = argv[1];
    const char* in_path  = argv[2];
    const char* out_path = argc > 3 ? argv[3] : "/tmp/siglip2_out.bin";
    int iters = argc > 4 ? atoi(argv[4]) : 1;
    if (iters < 1) iters = 1;

    setenv("ADSP_LIBRARY_PATH", "/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp", 1);
    enable_unsigned_pd();

    const char* qdir = getenv("QNN_LIB_DIR");
    if (!qdir) qdir = "/data/data/com.termux/files/home/qnn_v51";
    char s_sys[512], s_stub[512], s_htp[512];
    snprintf(s_sys,  sizeof(s_sys),  "%s/libQnnSystem.so",     qdir);
    snprintf(s_stub, sizeof(s_stub), "%s/libQnnHtpV68Stub.so", qdir);
    snprintf(s_htp,  sizeof(s_htp),  "%s/libQnnHtp.so",        qdir);
    void* h_sys = dlopen(s_sys, RTLD_NOW | RTLD_GLOBAL);
    if (!h_sys) { fprintf(stderr, "[qnn] sys: %s\n", dlerror()); return 2; }
    if (!dlopen(s_stub, RTLD_NOW | RTLD_GLOBAL)) { fprintf(stderr, "[qnn] stub: %s\n", dlerror()); return 2; }
    void* h_htp = dlopen(s_htp, RTLD_NOW | RTLD_GLOBAL);
    if (!h_htp) { fprintf(stderr, "[qnn] htp: %s\n", dlerror()); return 2; }

    typedef Qnn_ErrorHandle_t (*gp_t)(const QnnInterface_t***, uint32_t*);
    gp_t gp = (gp_t)dlsym(h_htp, "QnnInterface_getProviders");
    const QnnInterface_t** pv = NULL; uint32_t num = 0;
    Qnn_ErrorHandle_t e = gp(&pv, &num);
    if (e != QNN_SUCCESS || num == 0) { fprintf(stderr, "[qnn] gp err=%lu\n", (unsigned long)e); return 2; }
    QNN_INTERFACE_VER_TYPE qnn = pv[0]->QNN_INTERFACE_VER_NAME;
    fprintf(stderr, "[qnn] providers=%u name=%s\n", num, pv[0]->providerName ? pv[0]->providerName : "?");

    typedef Qnn_ErrorHandle_t (*sys_gp_t)(const QnnSystemInterface_t***, uint32_t*);
    sys_gp_t sys_gp = (sys_gp_t)dlsym(h_sys, "QnnSystemInterface_getProviders");
    const QnnSystemInterface_t** spv = NULL; uint32_t snum = 0;
    e = sys_gp(&spv, &snum);
    if (e != QNN_SUCCESS || snum == 0) { fprintf(stderr, "[sys] gp err=%lu\n", (unsigned long)e); return 2; }
    QNN_SYSTEM_INTERFACE_VER_TYPE sys = spv[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
    fprintf(stderr, "[sys] providers=%u\n", snum);

    Qnn_LogHandle_t log = NULL;
    e = qnn.logCreate(NULL, QNN_LOG_LEVEL_ERROR, &log);
    Qnn_BackendHandle_t be = NULL;
    e = qnn.backendCreate(log, NULL, &be);
    fprintf(stderr, "[backend] err=%lu\n", (unsigned long)e);
    if (e != QNN_SUCCESS) return 3;
    Qnn_DeviceHandle_t dev = NULL;
    e = qnn.deviceCreate(log, NULL, &dev);
    fprintf(stderr, "[device] err=%lu\n", (unsigned long)e);
    if (e != QNN_SUCCESS) return 3;

    size_t bin_sz = 0;
    void* bin_buf = read_file(bin_path, &bin_sz);
    if (!bin_buf) return 4;
    fprintf(stderr, "[ctx] sz=%zu\n", bin_sz);

    // ---- QnnSystemContext ----
    QnnSystemContext_Handle_t sysctx = NULL;
    e = sys.systemContextCreate(&sysctx);
    fprintf(stderr, "[sysctx] err=%lu\n", (unsigned long)e);
    if (e != QNN_SUCCESS) return 5;

    const QnnSystemContext_BinaryInfo_t* bin_info = NULL;
    Qnn_ContextBinarySize_t bin_info_size = 0;
    e = sys.systemContextGetBinaryInfo(sysctx, bin_buf, bin_sz, &bin_info, &bin_info_size);
    fprintf(stderr, "[binInfo] err=%lu size=%lu ver=%u\n",
            (unsigned long)e, (unsigned long)bin_info_size,
            bin_info ? (unsigned)bin_info->version : 0);
    if (e != QNN_SUCCESS || !bin_info) return 5;

    uint32_t ng = 0;
    const QnnSystemContext_GraphInfo_t* graphs = NULL;
    if (bin_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        ng = bin_info->contextBinaryInfoV3.numGraphs;
        graphs = bin_info->contextBinaryInfoV3.graphs;
    } else if (bin_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        ng = bin_info->contextBinaryInfoV2.numGraphs;
        graphs = bin_info->contextBinaryInfoV2.graphs;
    } else if (bin_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        ng = bin_info->contextBinaryInfoV1.numGraphs;
        graphs = bin_info->contextBinaryInfoV1.graphs;
    }
    fprintf(stderr, "[binInfo] graphs=%u\n", ng);

    // 遍历 graph 打名字
    for (uint32_t i = 0; i < ng && graphs; i++) {
        if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
            const QnnSystemContext_GraphInfoV3_t* g = &graphs[i].graphInfoV3;
            fprintf(stderr, "[graph %u] name=%s in=%u out=%u\n",
                    i, g->graphName ? g->graphName : "?", g->numGraphInputs, g->numGraphOutputs);
        }
    }

    Qnn_ContextHandle_t ctx = NULL;
    e = qnn.contextCreateFromBinary(be, dev, NULL, bin_buf, bin_sz, &ctx, NULL);
    fprintf(stderr, "[ctxCreate] err=%lu ctx=%p\n", (unsigned long)e, (void*)ctx);
    if (e != QNN_SUCCESS) return 6;

    Qnn_GraphHandle_t graph = NULL;
    e = qnn.graphRetrieve(ctx, "siglip2_model", &graph);
    fprintf(stderr, "[graph] err=%lu\n", (unsigned long)e);
    if (e != QNN_SUCCESS) return 6;

    // ---- 找匹配的 graphInfo ----
    const QnnSystemContext_GraphInfoV3_t* g3 = NULL;
    for (uint32_t i = 0; i < ng && graphs; i++) {
        if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
            const QnnSystemContext_GraphInfoV3_t* c = &graphs[i].graphInfoV3;
            if (c->graphName && strcmp(c->graphName, "siglip2_model") == 0) { g3 = c; break; }
        }
    }
    if (!g3) { fprintf(stderr, "[x] graph 'siglip2_model' not found in bin_info\n"); return 7; }

    Qnn_Tensor_t in_t, out_t;
    memset(&in_t, 0, sizeof(in_t));
    memset(&out_t, 0, sizeof(out_t));
    memcpy(&in_t,  &g3->graphInputs[0],  sizeof(Qnn_Tensor_t));
    memcpy(&out_t, &g3->graphOutputs[0], sizeof(Qnn_Tensor_t));
    dump_tensor("in_real",  &in_t);
    dump_tensor("out_real", &out_t);

    size_t in_read = 0;
    void* in_data = read_file(in_path, &in_read);
    if (!in_data) return 8;
    uint32_t in_bytes = in_t.v2.clientBuf.dataSize;
    if (in_bytes == 0) {
        uint32_t n = 1;
        for (uint32_t i = 0; i < in_t.v2.rank; i++) n *= in_t.v2.dimensions[i];
        in_bytes = n * 2;
    }
    fprintf(stderr, "[in] want=%u got=%zu\n", in_bytes, in_read);

    uint32_t out_bytes = out_t.v2.clientBuf.dataSize;
    if (out_bytes == 0) {
        uint32_t n = 1;
        for (uint32_t i = 0; i < out_t.v2.rank; i++) n *= out_t.v2.dimensions[i];
        out_bytes = n * 2;
    }
    fprintf(stderr, "[out] bytes=%u\n", out_bytes);

    void* out_data = calloc(1, out_bytes);
    if (!out_data) return 8;

    in_t.v2.clientBuf.data = in_data;
    in_t.v2.clientBuf.dataSize = in_bytes;
    out_t.v2.clientBuf.data = out_data;
    out_t.v2.clientBuf.dataSize = out_bytes;

    double a = now_ms();
    e = qnn.graphExecute(graph, &in_t, 1, &out_t, 1, NULL, NULL);
    double b = now_ms();
    fprintf(stderr, "[exec #1] err=%lu %.2f ms\n", (unsigned long)e, b - a);
    if (e != QNN_SUCCESS) return 9;

    double total = 0;
    for (int i = 0; i < iters; i++) {
        a = now_ms();
        e = qnn.graphExecute(graph, &in_t, 1, &out_t, 1, NULL, NULL);
        b = now_ms();
        if (e != QNN_SUCCESS) { fprintf(stderr, "[exec #%d] err=%lu\n", i+2, (unsigned long)e); break; }
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
