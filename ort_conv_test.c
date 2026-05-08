/*
 * Conv diagnostic test: runs 3 minimal Conv models on ORT,
 * prints output as hex for cross-platform comparison.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "onnxruntime_c_api.h"

const OrtApi* g_ort = NULL;

static void print_hex(const char* label, float* data, int n) {
    printf("%s (%d values):\n", label, n);
    for (int i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &data[i], sizeof(bits));
        printf("  [%2d] 0x%08x\n", i, bits);
    }
}

#define CHECK_STATUS(expr) do { \
    OrtStatus* status = (expr); \
    if (status != NULL) { \
        const char* msg = g_ort->GetErrorMessage(status); \
        fprintf(stderr, "FAIL: %s\n", msg); \
        g_ort->ReleaseStatus(status); \
        return -1; \
    } \
} while(0)

int main(int argc, char* argv[]) {
    const char* model_dir = ".";
    if (argc > 1) model_dir = argv[1];

    printf("conv diagnostic test starting...\n");
    printf("model_dir: %s\n", model_dir);

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) { fprintf(stderr, "Failed to get OrtApi\n"); return 1; }

    OrtEnv* env = NULL;
    CHECK_STATUS(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "conv_test", &env));

    /* ========== Model 1: conv1x1 (1,2,3,3) -> (1,3,3,3) ========== */
    {
        printf("\n===== conv1x1 (1x1 Conv, GEMM only) =====\n");
        const char* mpath = "/usr/bin/conv1x1.ort";
        if (argc > 1) {
            static char buf[256];
            snprintf(buf, sizeof(buf), "%s/conv1x1.ort", model_dir);
            mpath = buf;
        }

        OrtSessionOptions* opts = NULL;
        CHECK_STATUS(g_ort->CreateSessionOptions(&opts));
        CHECK_STATUS(g_ort->SetIntraOpNumThreads(opts, 1));
        CHECK_STATUS(g_ort->SetSessionGraphOptimizationLevel(opts, ORT_DISABLE_ALL));

        OrtSession* sess = NULL;
        CHECK_STATUS(g_ort->CreateSession(env, mpath, opts, &sess));
        printf("Model loaded\n");

        OrtMemoryInfo* mi = NULL;
        CHECK_STATUS(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));

        /* Deterministic input */
        int64_t shape[] = {1, 2, 3, 3};
        float data[18];
        for (int i = 0; i < 18; i++) data[i] = (float)((i % 17) - 8) * 0.1f;

        OrtValue* input = NULL;
        CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(
            mi, data, sizeof(data), shape, 4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input));

        OrtAllocator* alloc = NULL;
        CHECK_STATUS(g_ort->GetAllocatorWithDefaultOptions(&alloc));

        char* in_name = NULL;
        CHECK_STATUS(g_ort->SessionGetInputName(sess, 0, alloc, &in_name));
        char* out_name = NULL;
        CHECK_STATUS(g_ort->SessionGetOutputName(sess, 0, alloc, &out_name));

        const char* in_names[] = {in_name};
        const char* out_names[] = {out_name};

        print_hex("input", data, 18);

        OrtValue* output = NULL;
        CHECK_STATUS(g_ort->Run(sess, NULL, in_names, (const OrtValue* const*)&input, 1,
                                out_names, 1, &output));
        printf("Inference done\n");

        float* out_data = NULL;
        CHECK_STATUS(g_ort->GetTensorMutableData(output, (void**)&out_data));

        OrtTensorTypeAndShapeInfo* oi = NULL;
        CHECK_STATUS(g_ort->GetTensorTypeAndShape(output, &oi));
        size_t ndim;
        CHECK_STATUS(g_ort->GetDimensionsCount(oi, &ndim));
        int64_t od[8];
        CHECK_STATUS(g_ort->GetDimensions(oi, od, ndim));
        size_t total = 1;
        printf("Output shape: [");
        for (size_t d = 0; d < ndim; d++) {
            printf("%lld%s", (long long)od[d], d+1<ndim?",":"");
            total *= od[d];
        }
        printf("] (%zu elements)\n", total);

        print_hex("output", out_data, (int)total);
        printf("conv1x1 PASSED\n");

        g_ort->ReleaseValue(output);
        g_ort->ReleaseValue(input);
        g_ort->ReleaseSession(sess);
        g_ort->ReleaseSessionOptions(opts);
        g_ort->ReleaseMemoryInfo(mi);
    }

    /* ========== Model 2: conv3x3 (1,1,5,5) -> (1,1,3,3) ========== */
    {
        printf("\n===== conv3x3 (3x3 Conv, im2col+GEMM) =====\n");
        const char* mpath = "/usr/bin/conv3x3.ort";
        if (argc > 1) {
            static char buf[256];
            snprintf(buf, sizeof(buf), "%s/conv3x3.ort", model_dir);
            mpath = buf;
        }

        OrtSessionOptions* opts = NULL;
        CHECK_STATUS(g_ort->CreateSessionOptions(&opts));
        CHECK_STATUS(g_ort->SetIntraOpNumThreads(opts, 1));
        CHECK_STATUS(g_ort->SetSessionGraphOptimizationLevel(opts, ORT_DISABLE_ALL));

        OrtSession* sess = NULL;
        CHECK_STATUS(g_ort->CreateSession(env, mpath, opts, &sess));
        printf("Model loaded\n");

        OrtMemoryInfo* mi = NULL;
        CHECK_STATUS(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));

        int64_t shape[] = {1, 1, 5, 5};
        float data[25];
        for (int i = 0; i < 25; i++) data[i] = (float)((i % 17) - 8) * 0.1f;

        OrtValue* input = NULL;
        CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(
            mi, data, sizeof(data), shape, 4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input));

        OrtAllocator* alloc = NULL;
        CHECK_STATUS(g_ort->GetAllocatorWithDefaultOptions(&alloc));
        char* in_name = NULL;
        CHECK_STATUS(g_ort->SessionGetInputName(sess, 0, alloc, &in_name));
        char* out_name = NULL;
        CHECK_STATUS(g_ort->SessionGetOutputName(sess, 0, alloc, &out_name));
        const char* in_names[] = {in_name};
        const char* out_names[] = {out_name};

        print_hex("input", data, 25);

        OrtValue* output = NULL;
        CHECK_STATUS(g_ort->Run(sess, NULL, in_names, (const OrtValue* const*)&input, 1,
                                out_names, 1, &output));
        printf("Inference done\n");

        float* out_data = NULL;
        CHECK_STATUS(g_ort->GetTensorMutableData(output, (void**)&out_data));

        OrtTensorTypeAndShapeInfo* oi = NULL;
        CHECK_STATUS(g_ort->GetTensorTypeAndShape(output, &oi));
        size_t ndim;
        CHECK_STATUS(g_ort->GetDimensionsCount(oi, &ndim));
        int64_t od[8];
        CHECK_STATUS(g_ort->GetDimensions(oi, od, ndim));
        size_t total = 1;
        printf("Output shape: [");
        for (size_t d = 0; d < ndim; d++) {
            printf("%lld%s", (long long)od[d], d+1<ndim?",":"");
            total *= od[d];
        }
        printf("] (%zu elements)\n", total);

        print_hex("output", out_data, (int)total);
        printf("conv3x3 PASSED\n");

        g_ort->ReleaseValue(output);
        g_ort->ReleaseValue(input);
        g_ort->ReleaseSession(sess);
        g_ort->ReleaseSessionOptions(opts);
        g_ort->ReleaseMemoryInfo(mi);
    }

    /* ========== Model 3: conv_bn (1,1,5,5) -> (1,2,3,3) ========== */
    {
        printf("\n===== conv_bn (3x3 Conv + BatchNorm) =====\n");
        const char* mpath = "/usr/bin/conv_bn.ort";
        if (argc > 1) {
            static char buf[256];
            snprintf(buf, sizeof(buf), "%s/conv_bn.ort", model_dir);
            mpath = buf;
        }

        OrtSessionOptions* opts = NULL;
        CHECK_STATUS(g_ort->CreateSessionOptions(&opts));
        CHECK_STATUS(g_ort->SetIntraOpNumThreads(opts, 1));
        CHECK_STATUS(g_ort->SetSessionGraphOptimizationLevel(opts, ORT_DISABLE_ALL));

        OrtSession* sess = NULL;
        CHECK_STATUS(g_ort->CreateSession(env, mpath, opts, &sess));
        printf("Model loaded\n");

        OrtMemoryInfo* mi = NULL;
        CHECK_STATUS(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));

        int64_t shape[] = {1, 1, 5, 5};
        float data[25];
        for (int i = 0; i < 25; i++) data[i] = (float)((i % 17) - 8) * 0.1f;

        OrtValue* input = NULL;
        CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(
            mi, data, sizeof(data), shape, 4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input));

        OrtAllocator* alloc = NULL;
        CHECK_STATUS(g_ort->GetAllocatorWithDefaultOptions(&alloc));
        char* in_name = NULL;
        CHECK_STATUS(g_ort->SessionGetInputName(sess, 0, alloc, &in_name));
        char* out_name = NULL;
        CHECK_STATUS(g_ort->SessionGetOutputName(sess, 0, alloc, &out_name));
        const char* in_names[] = {in_name};
        const char* out_names[] = {out_name};

        print_hex("input", data, 25);

        OrtValue* output = NULL;
        CHECK_STATUS(g_ort->Run(sess, NULL, in_names, (const OrtValue* const*)&input, 1,
                                out_names, 1, &output));
        printf("Inference done\n");

        float* out_data = NULL;
        CHECK_STATUS(g_ort->GetTensorMutableData(output, (void**)&out_data));

        OrtTensorTypeAndShapeInfo* oi = NULL;
        CHECK_STATUS(g_ort->GetTensorTypeAndShape(output, &oi));
        size_t ndim;
        CHECK_STATUS(g_ort->GetDimensionsCount(oi, &ndim));
        int64_t od[8];
        CHECK_STATUS(g_ort->GetDimensions(oi, od, ndim));
        size_t total = 1;
        printf("Output shape: [");
        for (size_t d = 0; d < ndim; d++) {
            printf("%lld%s", (long long)od[d], d+1<ndim?",":"");
            total *= od[d];
        }
        printf("] (%zu elements)\n", total);

        print_hex("output", out_data, (int)total);
        printf("conv_bn PASSED\n");

        g_ort->ReleaseValue(output);
        g_ort->ReleaseValue(input);
        g_ort->ReleaseSession(sess);
        g_ort->ReleaseSessionOptions(opts);
        g_ort->ReleaseMemoryInfo(mi);
    }

    printf("\nALL TESTS PASSED\n");

    g_ort->ReleaseEnv(env);
    return 0;
}
