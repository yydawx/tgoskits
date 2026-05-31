/*
 * Multi-model TPU benchmark: runs 100 test cases on each model, prints results.
 * Usage: bench_multi <input_dir> <model1> <model2> ...
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cviruntime.h>

static void fp32_to_bf16(const float *src, uint16_t *dst, int count) {
    for (int i = 0; i < count; i++)
        dst[i] = ((const uint16_t *)src)[i * 2 + 1];
}
static void bf16_to_fp32(const uint16_t *src, float *dst, int count) {
    for (int i = 0; i < count; i++) {
        ((uint16_t *)dst)[i * 2 + 1] = src[i];
        ((uint16_t *)dst)[i * 2] = 0;
    }
}
static int read_raw(const char *path, void *buf, size_t expected) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return -1; }
    size_t n = fread(buf, 1, expected, f);
    fclose(f);
    if (n != expected) { fprintf(stderr, "short read %s\n", path); return -1; }
    return 0;
}

int main(int argc, char *argv[]) {
    const char *in_dir = "/tpu/test_inputs";
    if (argc < 2) { fprintf(stderr, "Usage: %s <input_dir> <model1> [model2...]\n", argv[0]); return 1; }
    in_dir = argv[1];

    char path[512];
    snprintf(path, sizeof(path), "%s/count.txt", in_dir);
    FILE *cf = fopen(path, "r");
    if (!cf) { fprintf(stderr, "missing %s\n", path); return 1; }
    int N = 0; fscanf(cf, "%d", &N); fclose(cf);
    fprintf(stderr, "Cases: %d, Models: %d\n", N, argc - 2);

    int img_count = 3 * 224 * 224;
    int state_count = 2;
    float *img_fp32 = (float *)malloc(img_count * sizeof(float));
    float *state_fp32 = (float *)malloc(state_count * sizeof(float));
    if (!img_fp32 || !state_fp32) return 1;

    /* Run each model */
    for (int m = 2; m < argc; m++) {
        const char *model_path = argv[m];
        fprintf(stderr, "\n=== Model: %s ===\n", model_path);

        CVI_MODEL_HANDLE model = NULL;
        CVI_RC ret = CVI_NN_RegisterModel(model_path, &model);
        if (ret != CVI_RC_SUCCESS) { fprintf(stderr, "FAIL: register %d\n", ret); continue; }

        CVI_TENSOR *inputs, *outputs;
        int32_t in_num, out_num;
        CVI_NN_GetInputOutputTensors(model, &inputs, &in_num, &outputs, &out_num);

        double total_ms = 0;
        for (int c = 0; c < N; c++) {
            snprintf(path, sizeof(path), "%s/images_%03d.bin", in_dir, c);
            if (read_raw(path, img_fp32, img_count * sizeof(float))) return 1;
            snprintf(path, sizeof(path), "%s/state_%03d.bin", in_dir, c);
            if (read_raw(path, state_fp32, state_count * sizeof(float))) return 1;

            for (int i = 0; i < in_num; i++) {
                CVI_TENSOR *t = &inputs[i];
                if (t->count == (size_t)img_count) {
                    if (t->fmt == CVI_FMT_BF16)
                        fp32_to_bf16(img_fp32, (uint16_t *)CVI_NN_TensorPtr(t), t->count);
                    else memcpy(CVI_NN_TensorPtr(t), img_fp32, t->mem_size);
                } else if (t->count == (size_t)state_count) {
                    if (t->fmt == CVI_FMT_BF16)
                        fp32_to_bf16(state_fp32, (uint16_t *)CVI_NN_TensorPtr(t), t->count);
                    else memcpy(CVI_NN_TensorPtr(t), state_fp32, t->mem_size);
                }
            }

            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            ret = CVI_NN_Forward(model, inputs, in_num, outputs, out_num);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
            total_ms += ms;

            size_t out_count = outputs[0].count;
            float *out_fp32 = (float *)malloc(out_count * sizeof(float));
            if (outputs[0].fmt == CVI_FMT_BF16)
                bf16_to_fp32((uint16_t *)CVI_NN_TensorPtr(&outputs[0]), out_fp32, out_count);
            else memcpy(out_fp32, CVI_NN_TensorPtr(&outputs[0]), out_count * sizeof(float));

            /* Print: M <model_idx> C<c> <ms> <v0> ... <v15> */
            printf("M%d C%d %.1f", m - 2, c, ms);
            for (size_t j = 0; j < out_count; j++) printf(" %.4f", out_fp32[j]);
            printf("\n");
            fflush(stdout);
            free(out_fp32);
        }
        fprintf(stderr, "  Avg: %.2f ms, Total: %.2f ms\n", total_ms / N, total_ms);
        /* Skip CVI_NN_CleanupModel — its C++ dtor triggers SIGILL.
           _exit() releases all resources cleanly without dtors. */
    }

    fprintf(stderr, "Done.\n");
    _exit(0);
}
