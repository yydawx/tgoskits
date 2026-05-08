#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include "onnxruntime_c_api.h"

static void print_heap(const char* label) {
    printf("[MEM-v4-8May] %s\n", label);
    void* brk = sbrk(0);
    printf("[BRK-v4] brk=%p\n", brk);
}

const OrtApi* g_ort = NULL;

static void print_float_hex(const char* label, float* data, int n) {
    printf("%s (hex):", label);
    for (int i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &data[i], sizeof(bits));
        printf(" 0x%08x", bits);
    }
    printf("\n");
}

static float hex_to_float(uint32_t bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

#define CHECK_STATUS(expr) do { \
    OrtStatus* status = (expr); \
    if (status != NULL) { \
        const char* msg = g_ort->GetErrorMessage(status); \
        fprintf(stderr, "Error: %s\n", msg); \
        g_ort->ReleaseStatus(status); \
        return 1; \
    } \
} while(0)

int main(int argc, char* argv[]) {
    const char* model_path = "/model.onnx";
    if (argc > 1) model_path = argv[1];

    printf("onnxruntime test starting...\n");
    print_heap("start");

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) { fprintf(stderr, "Failed to get OrtApi\n"); return 1; }
    printf("Got OrtApi v%d\n", ORT_API_VERSION);

    OrtEnv* env = NULL;
    CHECK_STATUS(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ort_test", &env));

    OrtSessionOptions* opts = NULL;
    CHECK_STATUS(g_ort->CreateSessionOptions(&opts));
    CHECK_STATUS(g_ort->SetIntraOpNumThreads(opts, 1));
    CHECK_STATUS(g_ort->SetSessionGraphOptimizationLevel(opts, ORT_DISABLE_ALL));
    CHECK_STATUS(g_ort->SetSessionExecutionMode(opts, ORT_SEQUENTIAL));
    CHECK_STATUS(g_ort->DisableMemPattern(opts));
    CHECK_STATUS(g_ort->DisableCpuMemArena(opts));
    printf("Session: sequential, mem_pattern=off, cpu_arena=off\n");

    OrtSession* session = NULL;
    printf("Loading model: %s\n", model_path);
    print_heap("before model load");
    CHECK_STATUS(g_ort->CreateSession(env, model_path, opts, &session));
    print_heap("after model load");
    printf("Model loaded\n");

    OrtAllocator* allocator = NULL;
    CHECK_STATUS(g_ort->GetAllocatorWithDefaultOptions(&allocator));

    size_t num_inputs = 0, num_outputs = 0;
    CHECK_STATUS(g_ort->SessionGetInputCount(session, &num_inputs));
    CHECK_STATUS(g_ort->SessionGetOutputCount(session, &num_outputs));
    printf("Inputs: %zu, Outputs: %zu\n", num_inputs, num_outputs);

    OrtMemoryInfo* mem_info = NULL;
    CHECK_STATUS(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

    /* Build inputs */
    const char** input_names = (const char**)malloc(num_inputs * sizeof(char*));
    OrtValue** input_tensors = (OrtValue**)malloc(num_inputs * sizeof(OrtValue*));
    float** input_datas = (float**)malloc(num_inputs * sizeof(float*));
    size_t* input_sizes = (size_t*)malloc(num_inputs * sizeof(size_t));

    for (size_t i = 0; i < num_inputs; i++) {
        char* name = NULL;
        CHECK_STATUS(g_ort->SessionGetInputName(session, i, allocator, &name));
        input_names[i] = name;

        OrtTypeInfo* ti = NULL;
        CHECK_STATUS(g_ort->SessionGetInputTypeInfo(session, i, &ti));
        const OrtTensorTypeAndShapeInfo* tsi = NULL;
        CHECK_STATUS(g_ort->CastTypeInfoToTensorInfo(ti, &tsi));

        size_t ndim = 0;
        CHECK_STATUS(g_ort->GetDimensionsCount(tsi, &ndim));
        int64_t* d = (int64_t*)malloc(ndim * sizeof(int64_t));
        CHECK_STATUS(g_ort->GetDimensions(tsi, d, ndim));

        size_t n = 1;
        printf("Input %zu '%s': [", i, name);
        for (size_t dd = 0; dd < ndim; dd++) {
            printf("%lld%s", (long long)d[dd], dd+1<ndim?",":"");
            n *= d[dd];
        }
        printf("] (%zu elements)\n", n);

        float* idata = (float*)calloc(n, sizeof(float));
        printf("  alloc addr=%p n=%zu\n", (void*)idata, n);

        for (size_t j = 0; j < n; j++) {
            int rem = (int)(j % 17);
            int sub = rem - 8;
            float fval = (float)sub * 0.1f;
            idata[j] = fval;
            if (j < 4) {
                uint32_t raw;
                memcpy(&raw, &fval, 4);
                printf("  fill[%zu]: j%%17=%d sub=%d val=0x%08x\n", j, rem, sub, raw);
            }
        }

        print_float_hex("  after_fill[0:4]", idata, n < 4 ? (int)n : 4);

        CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(
            mem_info, idata, n*sizeof(float), d, ndim,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensors[i]));

        print_float_hex("  after_tensor[0:4]", idata, n < 4 ? (int)n : 4);

        input_datas[i] = idata;
        input_sizes[i] = n;

        /* Also read back from tensor to verify */
        float* tensor_data = NULL;
        CHECK_STATUS(g_ort->GetTensorMutableData(input_tensors[i], (void**)&tensor_data));
        print_float_hex("  from_tensor[0:4]", tensor_data, n < 4 ? (int)n : 4);

        g_ort->ReleaseTypeInfo(ti);
        free(d);
    }

    /* Get output names */
    const char** output_names = (const char**)malloc(num_outputs * sizeof(char*));
    for (size_t i = 0; i < num_outputs; i++) {
        char* name = NULL;
        CHECK_STATUS(g_ort->SessionGetOutputName(session, i, allocator, &name));
        output_names[i] = name;
    }

    /* Run */
    OrtValue* output = NULL;
    print_heap("before inference");
    printf("Running inference...\n");
    CHECK_STATUS(g_ort->Run(session, NULL,
        input_names, (const OrtValue* const*)input_tensors, num_inputs,
        output_names, num_outputs, &output));
    printf("Inference done\n");
    print_heap("after inference");

    /* Check if input data was corrupted by inference */
    for (size_t i = 0; i < num_inputs; i++) {
        char label[64];
        snprintf(label, sizeof(label), "  post-infer input[%zu][0:4]", i);
        print_float_hex(label, input_datas[i], input_sizes[i] < 4 ? (int)input_sizes[i] : 4);
    }

    /* Read output */
    float* out_data = NULL;
    CHECK_STATUS(g_ort->GetTensorMutableData(output, (void**)&out_data));

    OrtTensorTypeAndShapeInfo* oi = NULL;
    CHECK_STATUS(g_ort->GetTensorTypeAndShape(output, &oi));
    size_t ondim = 0;
    CHECK_STATUS(g_ort->GetDimensionsCount(oi, &ondim));
    int64_t* od = (int64_t*)malloc(ondim * sizeof(int64_t));
    CHECK_STATUS(g_ort->GetDimensions(oi, od, ondim));
    size_t on = 1;
    printf("Output shape: [");
    for (size_t d = 0; d < ondim; d++) {
        printf("%lld%s", (long long)od[d], d+1<ondim?",":"");
        on *= od[d];
    }
    printf("] (%zu elements)\n", on);

    /* Print output as hex, one value per line for reliable parsing */
    printf("output (%zu values):\n", on);
    for (size_t i = 0; i < on && i < 256; i++) {
        uint32_t bits;
        memcpy(&bits, &out_data[i], sizeof(bits));
        printf("  [%3zu] 0x%08x\n", i, bits);
    }

    printf("TEST PASSED\n");

    /* cleanup */
    g_ort->ReleaseValue(output);
    for (size_t i = 0; i < num_inputs; i++) {
        g_ort->ReleaseValue(input_tensors[i]);
        free(input_datas[i]);
    }
    free(input_names); free(input_tensors); free(input_datas); free(input_sizes);
    free(output_names); free(od);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(opts);
    g_ort->ReleaseEnv(env);
    g_ort->ReleaseMemoryInfo(mem_info);
    return 0;
}
