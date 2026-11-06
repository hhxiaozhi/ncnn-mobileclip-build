// siglip2_infer.c v6 —— memcpy 真实 tensor 前 sizeof 字节，清零尾部，只传 pooler_output
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

#define T_SIZE 256

static void hex_dump(const char* tag, uint8_t* p, int n) {
    fprintf(stderr, "[%s] ", tag);
    for (int i = 0; i < n; i++) fprintf(stderr, "%02x", p[i]);
    fprintf(stderr, "\n");
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <bin> <in> <out> [graph] [out_idx] [iters]\n", argv[0]); return 1; }
    const char* bin_path = argv[1];
    const char* in_path  = argv[2];
    const char* out_path = argv[3];
    const char* graph_name = argc > 4 ? argv[4] : "siglip2_model";
    int out_idx = argc > 5 ? atoi(argv[5]) : 1;
    int iters = argc > 6 ? atoi(argv[6]) : 1;

    setenv("ADSP_LIBRARY_PATH", "/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp", 1);
    enable_unsigned_pd();

    const char* qdir = getenv("QNN_LIB_DIR");
    if (!qdir) qdir = "/data/data/com.termux/files/home/qnn_v51";
    char s_sys[512], s_stub[512], s_htp[512];
    snprintf(s_sys, sizeof(s_sys), "%s/libQnnSystem.so", qdir);
    snprintf(s_stub, sizeof(s_stub), "%s/libQnnHtpV68Stub.so", qdir);
    snprintf(s_htp, sizeof(s_htp), "%s/libQnnHtp.so", qdir);
    void* h_sys = dlopen(s_sys, RTLD_NOW|RTLD_GLOBAL);
    if (!h_sys) return 2;
    if (!dlopen(s_stub, RTLD_NOW|RTLD_GLOBAL)) return 2;
    void* h_htp = dlopen(s_htp, RTLD_NOW|RTLD_GLOBAL);
    if (!h_htp) return 2;

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
            if (graphs[i].graphInfoV3.graphName && strcmp(graphs[i].graphInfoV3.graphName, graph_name) == 0) {
                g3 = &graphs[i].graphInfoV3; break;
            }
        }
    }
    if (!g3) { fprintf(stderr, "[x] graph '%s' not found\n", graph_name); return 6; }

    Qnn_ContextHandle_t ctx = NULL;
    e = qnn.contextCreateFromBinary(be, dev, NULL, bin_buf, bin_sz, &ctx, NULL);
    if (e != QNN_SUCCESS) return 6;
    Qnn_GraphHandle_t graph = NULL;
    e = qnn.graphRetrieve(ctx, graph_name, &graph);
    if (e != QNN_SUCCESS) return 6;

    fprintf(stderr, "[graph] name=%s in=%u out=%u sizeof(Qnn_Tensor_t)=%zu\n",
            graph_name, g3->numGraphInputs, g3->numGraphOutputs, sizeof(Qnn_Tensor_t));

    // 输入：memcpy 真实的 sizeof(Qnn_Tensor_t) 字节，后面清零
    uint8_t in_raw[T_SIZE] = {0};
    memcpy(in_raw, &g3->graphInputs[0], sizeof(Qnn_Tensor_t));
    hex_dump("in_raw_orig", in_raw, sizeof(Qnn_Tensor_t));

    // 输出：memcpy 真实的 sizeof(Qnn_Tensor_t) 字节，后面清零
    uint8_t out_raw[T_SIZE] = {0};
    if (out_idx >= (int)g3->numGraphOutputs) { fprintf(stderr, "[x] out_idx out of range\n"); return 7; }
    memcpy(out_raw, &g3->graphOutputs[out_idx], sizeof(Qnn_Tensor_t));

    // 从 2.31 结构里提取关键字段（用标准 144 字节布局）
    Qnn_Tensor_t* in_231 = (Qnn_Tensor_t*)in_raw;
    Qnn_Tensor_t* out_231 = (Qnn_Tensor_t*)out_raw;
    fprintf(stderr, "[in_231] id=%u name=%s type=%d dtype=0x%x rank=%u datasize=%u\n",
            in_231->v2.id, in_231->v2.name ? in_231->v2.name : "?",
            (int)in_231->v2.type, (unsigned)in_231->v2.dataType,
            in_231->v2.rank, in_231->v2.clientBuf.dataSize);
    fprintf(stderr, "[out_231] id=%u name=%s type=%d dtype=0x%x rank=%u datasize=%u\n",
            out_231->v2.id, out_231->v2.name ? out_231->v2.name : "?",
            (int)out_231->v2.type, (unsigned)out_231->v2.dataType,
            out_231->v2.rank, out_231->v2.clientBuf.dataSize);

    // 计算输入字节数
    uint32_t in_elems = 1;
    for (uint32_t i = 0; i < in_231->v2.rank; i++) in_elems *= in_231->v2.dimensions[i];
    // 通过 dtype 推算元素字节
    uint32_t elem_bytes = 1;
    uint32_t dtype = (uint32_t)in_231->v2.dataType;
    // 0x216 = FLOAT_16 (2), 0x408 = UFIXED_8 (1), 0x2D1 = FLOAT_32 (4), 0x108/0x101 = UINT_8 (1)
    if (dtype == 0x216) elem_bytes = 2;
    else if (dtype == 0x2D1) elem_bytes = 4;
    else elem_bytes = 1;
    uint32_t in_bytes = in_elems * elem_bytes;

    uint32_t out_elems = 1;
    for (uint32_t i = 0; i < out_231->v2.rank; i++) out_elems *= out_231->v2.dimensions[i];
    uint32_t out_dtype = (uint32_t)out_231->v2.dataType;
    uint32_t out_elem_bytes = 1;
    if (out_dtype == 0x216) out_elem_bytes = 2;
    else if (out_dtype == 0x2D1) out_elem_bytes = 4;
    uint32_t out_bytes = out_elems * out_elem_bytes;

    fprintf(stderr, "[bytes] in=%u (elem=%u) out=%u (elem=%u)\n",
            in_bytes, elem_bytes, out_bytes, out_elem_bytes);

    size_t in_read = 0;
    void* in_data = read_file(in_path, &in_read);
    if (!in_data) return 8;
    if (in_read < in_bytes) { fprintf(stderr, "[x] input too small: want=%u got=%zu\n", in_bytes, in_read); return 8; }

    void* out_data = calloc(1, out_bytes);
    if (!out_data) return 8;

    // 覆盖 clientBuf 指针（2.31 布局，offset 104 = data, 112 = datasize）
    *(uint64_t*)(in_raw + 104) = (uint64_t)in_data;
    *(uint32_t*)(in_raw + 112) = in_bytes;
    *(uint64_t*)(out_raw + 104) = (uint64_t)out_data;
    *(uint32_t*)(out_raw + 112) = out_bytes;

    // 执行
    typedef Qnn_ErrorHandle_t (*gexec_t)(Qnn_GraphHandle_t, const void*, uint32_t, void*, uint32_t, void*, void*);
    gexec_t gexec = (gexec_t)qnn.graphExecute;

    // 用 malloc 分配 tensor buffer（排除栈问题）
    uint8_t* in_heap = (uint8_t*)malloc(T_SIZE); memcpy(in_heap, in_raw, T_SIZE);
    uint8_t* out_heap = (uint8_t*)malloc(T_SIZE); memcpy(out_heap, out_raw, T_SIZE);

    double a = now_ms();
    e = gexec(graph, in_heap, 1, out_heap, 1, NULL, NULL);
    double b = now_ms();
    fprintf(stderr, "[exec #1] err=%lu %.2f ms\n", (unsigned long)e, b - a);
    if (e != QNN_SUCCESS) {
        hex_dump("in_final_hex", in_heap, 128);
        return 9;
    }

    double total = 0;
    int ok = 0;
    for (int i = 0; i < iters; i++) {
        a = now_ms();
        e = gexec(graph, in_heap, 1, out_heap, 1, NULL, NULL);
        b = now_ms();
        if (e != QNN_SUCCESS) break;
        total += (b - a); ok++;
    }
    if (ok > 0) fprintf(stderr, "[exec avg %d] %.2f ms\n", ok, total / ok);

    FILE* of = fopen(out_path, "wb");
    if (of) { fwrite(out_data, 1, out_bytes, of); fclose(of); fprintf(stderr, "[out] %s %u bytes\n", out_path, out_bytes); }

    uint16_t* o16 = (uint16_t*)out_data;
    fprintf(stderr, "[out] first 8: ");
    for (int i = 0; i < 8 && (uint32_t)(i*2) < out_bytes; i++) fprintf(stderr, "0x%04x ", o16[i]);
    fprintf(stderr, "\n");

    free(in_data); free(out_data); free(bin_buf); free(in_heap); free(out_heap);
    return 0;
}
