// siglip2_infer_verbose.c — 带 QNN profile 的版本
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

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void log_cb(const char* fmt, QnnLog_Level_t level, uint64_t ts, va_list args) {
    const char* lv = "?";
    switch(level) {
        case QNN_LOG_LEVEL_ERROR: lv = "E"; break;
        case QNN_LOG_LEVEL_WARN: lv = "W"; break;
        case QNN_LOG_LEVEL_INFO: lv = "I"; break;
        case QNN_LOG_LEVEL_DEBUG: lv = "D"; break;
        case QNN_LOG_LEVEL_VERBOSE: lv = "V"; break;
        default: break;
    }
    fprintf(stderr, "[QNN-%s] ", lv);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
}

#define O_BUF   104
#define O_BUFSZ 112

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <bin> <in> <out_prefix>\n", argv[0]); return 1; }
    const char* bin_path = argv[1];
    const char* in_path = argv[2];
    const char* out_prefix = argv[3];

    setenv("ADSP_LIBRARY_PATH", "/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp", 1);
    enable_unsigned_pd();

    const char* qdir = getenv("QNN_LIB_DIR");
    if (!qdir) qdir = "/data/data/com.termux/files/home/qnn_v51";
    char s_sys[512], s_stub[512], s_htp[512];
    snprintf(s_sys, sizeof(s_sys), "%s/libQnnSystem.so", qdir);
    snprintf(s_stub, sizeof(s_stub), "%s/libQnnHtpV68Stub.so", qdir);
    snprintf(s_htp, sizeof(s_htp), "%s/libQnnHtp.so", qdir);
    void* h_sys = dlopen(s_sys, RTLD_NOW|RTLD_GLOBAL);
    void* h_htp = dlopen(s_htp, RTLD_NOW|RTLD_GLOBAL);
    dlopen(s_stub, RTLD_NOW|RTLD_GLOBAL);
    if (!h_sys || !h_htp) return 2;

    typedef Qnn_ErrorHandle_t (*gp_t)(const QnnInterface_t***, uint32_t*);
    gp_t gp = (gp_t)dlsym(h_htp, "QnnInterface_getProviders");
    const QnnInterface_t** pv = NULL; uint32_t num = 0;
    gp(&pv, &num);
    QNN_INTERFACE_VER_TYPE qnn = pv[0]->QNN_INTERFACE_VER_NAME;

    // 日志级别 VERBOSE
    Qnn_LogHandle_t log = NULL;
    qnn.logCreate(NULL, QNN_LOG_LEVEL_VERBOSE, &log);

    Qnn_BackendHandle_t be = NULL;
    qnn.backendCreate(log, NULL, &be);
    Qnn_DeviceHandle_t dev = NULL;
    qnn.deviceCreate(log, NULL, &dev);

    size_t bin_sz = 0;
    void* bin_buf = read_file(bin_path, &bin_sz);

    typedef Qnn_ErrorHandle_t (*sys_gp_t)(const QnnSystemInterface_t***, uint32_t*);
    sys_gp_t sys_gp = (sys_gp_t)dlsym(h_sys, "QnnSystemInterface_getProviders");
    const QnnSystemInterface_t** spv = NULL; uint32_t snum = 0;
    sys_gp(&spv, &snum);
    QNN_SYSTEM_INTERFACE_VER_TYPE sys = spv[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
    QnnSystemContext_Handle_t sysctx = NULL;
    sys.systemContextCreate(&sysctx);
    const QnnSystemContext_BinaryInfo_t* bi = NULL;
    Qnn_ContextBinarySize_t bi_sz = 0;
    sys.systemContextGetBinaryInfo(sysctx, bin_buf, bin_sz, &bi, &bi_sz);

    uint32_t ng = 0; const QnnSystemContext_GraphInfo_t* graphs = NULL;
    if (bi->version == 3) { ng = bi->contextBinaryInfoV3.numGraphs; graphs = bi->contextBinaryInfoV3.graphs; }
    const QnnSystemContext_GraphInfoV3_t* g3 = NULL;
    for (uint32_t i = 0; i < ng && graphs; i++) {
        if (graphs[i].version == 3 && graphs[i].graphInfoV3.graphName &&
            strcmp(graphs[i].graphInfoV3.graphName, "siglip2_model") == 0) {
            g3 = &graphs[i].graphInfoV3; break;
        }
    }
    if (!g3) return 6;
    uint32_t nout = g3->numGraphOutputs;

    Qnn_ContextHandle_t ctx = NULL;
    qnn.contextCreateFromBinary(be, dev, NULL, bin_buf, bin_sz, &ctx, NULL);
    Qnn_GraphHandle_t graph = NULL;
    qnn.graphRetrieve(ctx, "siglip2_model", &graph);

    uint8_t in_real[144] = {0};
    memcpy(in_real, (uint8_t*)g3->graphInputs, 144);
    uint32_t in_rank = *(uint32_t*)(in_real + 80);
    uint32_t* in_dims = *(uint32_t**)(in_real + 88);
    uint32_t in_elems = 1;
    for (uint32_t i = 0; i < in_rank; i++) in_elems *= in_dims[i];
    uint32_t in_bytes = in_elems * 2;

    size_t in_read = 0;
    void* in_data = read_file(in_path, &in_read);

    uint8_t out_real[8][144];
    uint32_t out_bytes_arr[8] = {0};
    void* out_data_arr[8] = {NULL};
    for (uint32_t i = 0; i < nout; i++) {
        memcpy(out_real[i], (uint8_t*)g3->graphOutputs + 144*i, 144);
        uint32_t rk = *(uint32_t*)(out_real[i] + 80);
        uint32_t* dm = *(uint32_t**)(out_real[i] + 88);
        uint32_t ne = 1;
        for (uint32_t j = 0; j < rk; j++) ne *= dm[j];
        out_bytes_arr[i] = ne * 2;
        out_data_arr[i] = calloc(1, out_bytes_arr[i]);
    }

    // profile 句柄
    Qnn_ProfileHandle_t profile = NULL;
    typedef Qnn_ErrorHandle_t (*prof_create_t)(Qnn_BackendHandle_t, Qnn_ProfileHandle_t*);
    prof_create_t prof_create = (prof_create_t)qnn.profileCreate;
    if (prof_create) prof_create(be, &profile);

    Qnn_Tensor_t in_t;
    memcpy(&in_t, in_real, sizeof(Qnn_Tensor_t));
    in_t.v2.clientBuf.data = in_data;
    in_t.v2.clientBuf.dataSize = in_bytes;

    Qnn_Tensor_t out_t[8];
    for (uint32_t i = 0; i < nout; i++) {
        memcpy(&out_t[i], out_real[i], sizeof(Qnn_Tensor_t));
        out_t[i].v2.clientBuf.data = out_data_arr[i];
        out_t[i].v2.clientBuf.dataSize = out_bytes_arr[i];
    }

    typedef Qnn_ErrorHandle_t (*gexec_t)(Qnn_GraphHandle_t, const void*, uint32_t, void*, uint32_t, void*, void*);
    gexec_t gexec = (gexec_t)qnn.graphExecute;

    fprintf(stderr, "[BEFORE EXEC]\n");
    double a = now_ms();
    Qnn_ErrorHandle_t e = gexec(graph, &in_t, 1, out_t, nout, profile, NULL);
    double b = now_ms();
    fprintf(stderr, "[EXEC] err=%lu  %.2f ms\n", (unsigned long)e, b - a);

    // 提取 profile
    if (profile) {
        typedef Qnn_ErrorHandle_t (*prof_get_t)(Qnn_ProfileHandle_t, Qnn_ProfileEvent_t**, uint32_t*);
        prof_get_t prof_get = (prof_get_t)qnn.profileGetEvents;
        if (prof_get) {
            Qnn_ProfileEvent_t* events = NULL;
            uint32_t ne = 0;
            if (prof_get(profile, &events, &ne) == QNN_SUCCESS) {
                fprintf(stderr, "[PROFILE] %u events\n", ne);
                for (uint32_t i = 0; i < ne && i < 50; i++) {
                    fprintf(stderr, "  event[%u] type=%d unit=%d value=%lu\n",
                        i, (int)events[i].eventType, (int)events[i].unit, (unsigned long)events[i].value);
                }
                typedef Qnn_ErrorHandle_t (*prof_free_t)(Qnn_ProfileEvent_t*);
                prof_free_t pf = (prof_free_t)qnn.profileFreeEvents;
                if (pf) pf(events);
            }
        }
    }

    for (uint32_t i = 0; i < nout; i++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%u.bin", out_prefix, i);
        FILE* of = fopen(path, "wb");
        if (of) { fwrite(out_data_arr[i], 1, out_bytes_arr[i], of); fclose(of); }
    }

    free(in_data);
    for (uint32_t i = 0; i < nout; i++) free(out_data_arr[i]);
    free(bin_buf);
    fprintf(stderr, "[DONE]\n");
    return 0;
}
