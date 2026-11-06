// siglip2_infer.c
// SigLIP2 So400M fp16 QNN 推理 (HTP v68)
// 编译: aarch64-linux-android24-clang (NDK) + QAIRT 2.31 头文件
// 运行: 容器/Android bionic 环境，dlopen libQnnHtp.so (QNN 2.51)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <time.h>

#include "QnnInterface.h"
#include "QnnTypes.h"

// ---- Unsigned PD ----
struct remote_rpc_control_unsigned_module { int32_t domain; int32_t enable; };
#define DSPRPC_CONTROL_UNSIGNED_MODULE 2

static int enable_unsigned_pd(void) {
    void* h = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) { fprintf(stderr, "[pd] dlopen fail: %s\n", dlerror()); return -1; }
    typedef int (*rsc_t)(uint32_t, void*, uint32_t);
    rsc_t rsc = (rsc_t)dlsym(h, "remote_session_control");
    if (!rsc) { fprintf(stderr, "[pd] no remote_session_control\n"); return -1; }
    struct remote_rpc_control_unsigned_module um = {3, 1};
    int r = rsc(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof(um));
    fprintf(stderr, "[pd] r=%d\n", r);
    return r;
}

// ---- 文件 ----
static void* read_file(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc(sz);
    if (!p) { fclose(f); return NULL; }
    if (fread(p, 1, sz, f) != (size_t)sz) { free(p); fclose(f); return NULL; }
    fclose(f); *out_size = (size_t)sz; return p;
}

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static Qnn_Tensor_t make_tensor_fp16(uint32_t id, const char* name, Qnn_TensorType_t type,
                                     uint32_t* dims, uint32_t rank,
                                     void* buf, uint32_t bytes) {
    Qnn_Tensor_t t;
    memset(&t, 0, sizeof(t));
    t.version = QNN_TENSOR_VERSION_2;
    t.v2.id = id;
    t.v2.name = name;
    t.v2.type = type;
    t.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
    t.v2.dataType = QNN_DATATYPE_FLOAT_16;
    t.v2.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    t.v2.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    t.v2.rank = rank;
    t.v2.dimensions = dims;
    t.v2.memType = QNN_TENSORMEMTYPE_RAW;
    t.v2.clientBuf.data = buf;
    t.v2.clientBuf.dataSize = bytes;
    return t;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <ctx_bin> <input_fp16_nhwc_1x384x384x3> [output_fp16_1x1152] [iters]\n", argv[0]);
        return 1;
    }
    const char* bin_path = argv[1];
    const char* in_path  = argv[2];
    const char* out_path = argc > 3 ? argv[3] : "siglip2_out.bin";
    int iters = argc > 4 ? atoi(argv[4]) : 1;
    if (iters < 1) iters = 1;

    setenv("ADSP_LIBRARY_PATH", "/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp", 1);
    enable_unsigned_pd();

    // ---- 加载 QNN 库 ----
    const char* qdir = getenv("QNN_LIB_DIR");
    if (!qdir) qdir = "/data/data/com.termux/files/home/qnn_v51";
    char s_sys[512], s_stub[512], s_htp[512];
    snprintf(s_sys,  sizeof(s_sys),  "%s/libQnnSystem.so",       qdir);
    snprintf(s_stub, sizeof(s_stub), "%s/libQnnHtpV68Stub.so",   qdir);
    snprintf(s_htp,  sizeof(s_htp),  "%s/libQnnHtp.so",          qdir);
    if (!dlopen(s_sys,  RTLD_NOW | RTLD_GLOBAL)) { fprintf(stderr, "[qnn] sys: %s\n",  dlerror()); return 2; }
    if (!dlopen(s_stub, RTLD_NOW | RTLD_GLOBAL)) { fprintf(stderr, "[qnn] stub: %s\n", dlerror()); return 2; }
    void* h_htp = dlopen(s_htp, RTLD_NOW | RTLD_GLOBAL);
    if (!h_htp) { fprintf(stderr, "[qnn] htp: %s\n", dlerror()); return 2; }

    typedef Qnn_ErrorHandle_t (*gp_t)(const QnnInterface_t***, uint32_t*);
    gp_t gp = (gp_t)dlsym(h_htp, "QnnInterface_getProviders");
    if (!gp) { fprintf(stderr, "[qnn] no getProviders: %s\n", dlerror()); return 2; }
    const QnnInterface_t** providers = NULL;
    uint32_t num = 0;
    Qnn_ErrorHandle_t e = gp(&providers, &num);
    if (e != QNN_SUCCESS || num == 0 || !providers) {
        fprintf(stderr, "[qnn] getProviders err=%lu num=%u\n", (unsigned long)e, num); return 2;
    }
    QNN_INTERFACE_VER_TYPE qnn = providers[0]->QNN_INTERFACE_VER_NAME;
    fprintf(stderr, "[qnn] providers=%u providerName=%s\n",
            num, providers[0]->providerName ? providers[0]->providerName : "?");

    // ---- Log / Backend / Device ----
    Qnn_LogHandle_t log = NULL;
    e = qnn.logCreate(NULL, QNN_LOG_LEVEL_ERROR, &log);
    fprintf(stderr, "[logCreate] err=%lu\n", (unsigned long)e);

    Qnn_BackendHandle_t be = NULL;
    e = qnn.backendCreate(log, NULL, &be);
    fprintf(stderr, "[backendCreate] err=%lu be=%p\n", (unsigned long)e, (void*)be);
    if (e != QNN_SUCCESS) return 3;

    Qnn_DeviceHandle_t dev = NULL;
    e = qnn.deviceCreate(log, NULL, &dev);
    fprintf(stderr, "[deviceCreate] err=%lu dev=%p\n", (unsigned long)e, (void*)dev);
    if (e != QNN_SUCCESS) return 3;

    // ---- 加载 context binary ----
    size_t bin_sz = 0;
    void* bin_buf = read_file(bin_path, &bin_sz);
    if (!bin_buf) return 4;
    fprintf(stderr, "[ctx] bin=%s sz=%zu\n", bin_path, bin_sz);

    Qnn_ContextHandle_t ctx = NULL;
    e = qnn.contextCreateFromBinary(be, dev, NULL, bin_buf, bin_sz, &ctx, NULL);
    fprintf(stderr, "[ctxCreate] err=%lu ctx=%p\n", (unsigned long)e, (void*)ctx);
    if (e != QNN_SUCCESS) return 5;

    // ---- graph ----
    Qnn_GraphHandle_t graph = NULL;
    e = qnn.graphRetrieve(ctx, "siglip2_model", &graph);
    fprintf(stderr, "[graphRetrieve] err=%lu graph=%p\n", (unsigned long)e, (void*)graph);
    if (e != QNN_SUCCESS) return 6;

    // ---- tensors ----
    static uint32_t in_dims[4]  = {1, 384, 384, 3};
    static uint32_t out_dims[2] = {1, 1152};
    const size_t in_bytes  = (size_t)1 * 384 * 384 * 3 * 2;
    const size_t out_bytes = (size_t)1152 * 2;

    size_t in_read = 0;
    void* in_data = read_file(in_path, &in_read);
    if (!in_data) return 7;
    if (in_read != in_bytes) {
        fprintf(stderr, "[in] size mismatch want=%zu got=%zu\n", in_bytes, in_read); return 7;
    }
    void* out_data = malloc(out_bytes);
    if (!out_data) return 7;
    memset(out_data, 0, out_bytes);

    Qnn_Tensor_t in_t  = make_tensor_fp16(0, "pixel_values",  QNN_TENSOR_TYPE_APP_WRITE, in_dims,  4, in_data,  (uint32_t)in_bytes);
    Qnn_Tensor_t out_t = make_tensor_fp16(0, "pooler_output", QNN_TENSOR_TYPE_APP_READ,  out_dims, 2, out_data, (uint32_t)out_bytes);

    // ---- 首次推理 ----
    double a = now_ms();
    e = qnn.graphExecute(graph, &in_t, 1, &out_t, 1, NULL, NULL);
    double b = now_ms();
    fprintf(stderr, "[exec #1] err=%lu time=%.2f ms\n", (unsigned long)e, b - a);
    if (e != QNN_SUCCESS) return 8;

    // ---- 循环基准 ----
    double total = 0;
    for (int i = 0; i < iters; i++) {
        a = now_ms();
        e = qnn.graphExecute(graph, &in_t, 1, &out_t, 1, NULL, NULL);
        b = now_ms();
        if (e != QNN_SUCCESS) { fprintf(stderr, "[exec #%d] err=%lu\n", i+2, (unsigned long)e); break; }
        total += (b - a);
    }
    if (iters > 0) fprintf(stderr, "[exec avg %d] %.2f ms\n", iters, total / iters);

    // ---- 保存输出 ----
    FILE* of = fopen(out_path, "wb");
    if (of) { fwrite(out_data, 1, out_bytes, of); fclose(of); fprintf(stderr, "[out] wrote %s (%zu bytes)\n", out_path, out_bytes); }

    // ---- sanity 打印前 8 个 fp16 原值 ----
    uint16_t* o16 = (uint16_t*)out_data;
    fprintf(stderr, "[out] first 8 fp16 hex: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "0x%04x ", o16[i]);
    fprintf(stderr, "\n");

    free(in_data); free(out_data); free(bin_buf);
    return 0;
}
