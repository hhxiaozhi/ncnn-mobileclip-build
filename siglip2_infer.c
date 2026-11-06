// siglip2_infer.c v5 —— 手工构造 256B tensor（避开 2.31/2.51 布局差异）+ 双输出
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

// QNN 2.51 runtime 的 256 字节 tensor 布局（实测，非头文件）
#define T_SIZE    256
#define T_VER     0
#define T_ID      8
#define T_NAME    16
#define T_TYPE    24
#define T_DFMT    28
#define T_DTYPE   32
#define T_QDEF    40
#define T_QENC    44
#define T_RANK    80
#define T_DIMS    88
#define T_MEM     96
#define T_BUF     104
#define T_BUFSZ   112

// 手工构造一个干净的 256 字节 tensor
static void build_tensor(uint8_t* t, uint32_t id, const char* name,
                          uint32_t type, uint32_t dtype,
                          uint32_t* dims, uint32_t rank,
                          void* buf, uint32_t bytes) {
    memset(t, 0, T_SIZE);
    *(uint32_t*)(t + T_VER) = 2;                     // QNN_TENSOR_VERSION_2
    *(uint32_t*)(t + T_ID) = id;
    *(uint64_t*)(t + T_NAME) = (uint64_t)name;
    *(uint32_t*)(t + T_TYPE) = type;                 // 0=APP_WRITE 1=APP_READ
    *(uint32_t*)(t + T_DFMT) = 0;                    // DENSE
    *(uint32_t*)(t + T_DTYPE) = dtype;               // 0x216 = FLOAT_16
    *(uint32_t*)(t + T_QDEF) = 0x7FFFFFFF;           // DEFINITION_UNDEFINED
    *(uint32_t*)(t + T_QENC) = 0x7FFFFFFF;           // QUANTIZATION_ENCODING_UNDEFINED
    *(uint32_t*)(t + T_RANK) = rank;
    *(uint64_t*)(t + T_DIMS) = (uint64_t)dims;
    *(uint32_t*)(t + T_MEM) = 0;                     // RAW
    *(uint64_t*)(t + T_BUF) = (uint64_t)buf;
    *(uint32_t*)(t + T_BUFSZ) = bytes;
    // 120-255 全 0：isDynamicDimensions=null, sparseParams=UNDEFINED, isProduced=0
}

// 从 2.31 解析出的真实 tensor 读关键字段（避免写错 id/dtype）
#define R_ID(p)    (*(uint32_t*)((uint8_t*)(p) + T_ID))
#define R_DTYPE(p) (*(uint32_t*)((uint8_t*)(p) + T_DTYPE))
#define R_RANK(p)  (*(uint32_t*)((uint8_t*)(p) + T_RANK))
#define R_DIMS(p)  (*(uint32_t**)((uint8_t*)(p) + T_DIMS))

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <bin> <in> [out_pooler] [iters]\n", argv[0]); return 1; }
    const char* bin_path = argv[1];
    const char* in_path  = argv[2];
    const char* out_path = argc > 3 ? argv[3] : "/tmp/siglip2_pooler.bin";
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
    sys.systemContextCreate(&sysctx);
    const QnnSystemContext_BinaryInfo_t* bi = NULL;
    Qnn_ContextBinarySize_t bi_sz = 0;
    e = sys.systemContextGetBinaryInfo(sysctx, bin_buf, bin_sz, &bi, &bi_sz);
    if (e != QNN_SUCCESS || !bi) { fprintf(stderr, "[bi] %lu\n", (unsigned long)e); return 5; }

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

    Qnn_ContextHandle_t ctx = NULL;
    e = qnn.contextCreateFromBinary(be, dev, NULL, bin_buf, bin_sz, &ctx, NULL);
    if (e != QNN_SUCCESS) return 6;
    Qnn_GraphHandle_t graph = NULL;
    e = qnn.graphRetrieve(ctx, "siglip2_model", &graph);
    if (e != QNN_SUCCESS) return 6;

    fprintf(stderr, "[graph] in=%u out=%u\n", g3->numGraphInputs, g3->numGraphOutputs);

    // ---- 读真实 id/dtype/dims ----
    uint32_t in_id    = R_ID(&g3->graphInputs[0]);
    uint32_t in_dtype = R_DTYPE(&g3->graphInputs[0]);
    uint32_t* in_dims_real = R_DIMS(&g3->graphInputs[0]);
    uint32_t in_rank = R_RANK(&g3->graphInputs[0]);

    uint32_t out0_id = R_ID(&g3->graphOutputs[0]);
    uint32_t out0_dtype = R_DTYPE(&g3->graphOutputs[0]);
    uint32_t* out0_dims_real = R_DIMS(&g3->graphOutputs[0]);
    uint32_t out0_rank = R_RANK(&g3->graphOutputs[0]);

    uint32_t out1_id = R_ID(&g3->graphOutputs[1]);
    uint32_t out1_dtype = R_DTYPE(&g3->graphOutputs[1]);
    uint32_t* out1_dims_real = R_DIMS(&g3->graphOutputs[1]);
    uint32_t out1_rank = R_RANK(&g3->graphOutputs[1]);

    fprintf(stderr, "[in] id=%u dtype=0x%x rank=%u dims=[", in_id, in_dtype, in_rank);
    for (uint32_t i = 0; i < in_rank; i++) fprintf(stderr, "%u%s", in_dims_real[i], i+1<in_rank?",":"");
    fprintf(stderr, "]\n");
    fprintf(stderr, "[out0] id=%u dtype=0x%x rank=%u dims=[", out0_id, out0_dtype, out0_rank);
    for (uint32_t i = 0; i < out0_rank; i++) fprintf(stderr, "%u%s", out0_dims_real[i], i+1<out0_rank?",":"");
    fprintf(stderr, "]\n");
    fprintf(stderr, "[out1] id=%u dtype=0x%x rank=%u dims=[", out1_id, out1_dtype, out1_rank);
    for (uint32_t i = 0; i < out1_rank; i++) fprintf(stderr, "%u%s", out1_dims_real[i], i+1<out1_rank?",":"");
    fprintf(stderr, "]\n");

    // ---- 计算字节数 ----
    uint32_t in_elems = 1;
    for (uint32_t i = 0; i < in_rank; i++) in_elems *= in_dims_real[i];
    uint32_t in_bytes = in_elems * 2;

    uint32_t out0_elems = 1;
    for (uint32_t i = 0; i < out0_rank; i++) out0_elems *= out0_dims_real[i];
    uint32_t out0_bytes = out0_elems * 2;

    uint32_t out1_elems = 1;
    for (uint32_t i = 0; i < out1_rank; i++) out1_elems *= out1_dims_real[i];
    uint32_t out1_bytes = out1_elems * 2;

    fprintf(stderr, "[bytes] in=%u out0=%u out1=%u\n", in_bytes, out0_bytes, out1_bytes);

    // ---- 准备 buffer ----
    size_t in_read = 0;
    void* in_data = read_file(in_path, &in_read);
    if (!in_data) return 7;
    if (in_read != in_bytes) { fprintf(stderr, "[in] size mismatch: want=%u got=%zu\n", in_bytes, in_read); return 7; }

    void* out0_data = calloc(1, out0_bytes);
    void* out1_data = calloc(1, out1_bytes);
    if (!out0_data || !out1_data) return 7;

    // ---- 手工构造干净的 256 字节 tensor ----
    uint8_t in_t[T_SIZE], out0_t[T_SIZE], out1_t[T_SIZE];
    build_tensor(in_t,   in_id,   "pixel_values",     0, in_dtype,   in_dims_real,   in_rank,   in_data,   in_bytes);
    build_tensor(out0_t, out0_id, "last_hidden_state", 1, out0_dtype, out0_dims_real, out0_rank, out0_data, out0_bytes);
    build_tensor(out1_t, out1_id, "pooler_output",     1, out1_dtype, out1_dims_real, out1_rank, out1_data, out1_bytes);

    // 打印构造后的关键字段
    fprintf(stderr, "[built in] ver=%u id=%u type=%u dtype=0x%x rank=%u datasize=%u\n",
            *(uint32_t*)(in_t+T_VER), *(uint32_t*)(in_t+T_ID),
            *(uint32_t*)(in_t+T_TYPE), *(uint32_t*)(in_t+T_DTYPE),
            *(uint32_t*)(in_t+T_RANK), *(uint32_t*)(in_t+T_BUFSZ));

    // ---- 双输出调用（用 cast 绕过 2.31 头文件的 144 字节类型限制）----
    typedef Qnn_ErrorHandle_t (*gexec_t)(Qnn_GraphHandle_t, const void*, uint32_t, void*, uint32_t, void*, void*);
    gexec_t gexec = (gexec_t)qnn.graphExecute;

    Qnn_Tensor_t out_arr[2];  // 仅用于占位，实际传给 gexec 的是 out0_t/out1_t
    memset(out_arr, 0, sizeof(out_arr));
    memcpy(&out_arr[0], out0_t, sizeof(Qnn_Tensor_t));
    memcpy(&out_arr[1], out1_t, sizeof(Qnn_Tensor_t));

    // 双输出必须连在一起（连续 512 字节）
    uint8_t out_pair[512];
    memcpy(out_pair, out0_t, T_SIZE);
    memcpy(out_pair + T_SIZE, out1_t, T_SIZE);

    double a = now_ms();
    e = gexec(graph, in_t, 1, out_pair, 2, NULL, NULL);
    double b = now_ms();
    fprintf(stderr, "[exec #1] err=%lu %.2f ms\n", (unsigned long)e, b - a);
    if (e != QNN_SUCCESS) return 9;

    // 循环
    double total = 0;
    int ok = 0;
    for (int i = 0; i < iters; i++) {
        a = now_ms();
        e = gexec(graph, in_t, 1, out_pair, 2, NULL, NULL);
        b = now_ms();
        if (e != QNN_SUCCESS) { fprintf(stderr, "[exec #%d] err=%lu\n", i+2, (unsigned long)e); break; }
        total += (b - a); ok++;
    }
    if (ok > 0) fprintf(stderr, "[exec avg %d] %.2f ms\n", ok, total / ok);

    // ---- 保存 pooler_output（通常是 [1,1152]，适合做 embedding）----
    FILE* of = fopen(out_path, "wb");
    if (of) { fwrite(out1_data, 1, out1_bytes, of); fclose(of); fprintf(stderr, "[out] %s %u bytes (pooler_output)\n", out_path, out1_bytes); }

    // 打印 pooler 前 8 个 fp16
    uint16_t* o16 = (uint16_t*)out1_data;
    fprintf(stderr, "[pooler] first 8 fp16: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "0x%04x ", o16[i]);
    fprintf(stderr, "\n");

    // 也保存 last_hidden_state 便于检查
    char path2[1024];
    snprintf(path2, sizeof(path2), "%s.last_hidden", out_path);
    FILE* of2 = fopen(path2, "wb");
    if (of2) { fwrite(out0_data, 1, out0_bytes, of2); fclose(of2); fprintf(stderr, "[out] %s %u bytes (last_hidden_state)\n", path2, out0_bytes); }

    free(in_data); free(out0_data); free(out1_data); free(bin_buf);
    return 0;
}
