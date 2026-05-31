/*
 * ACT model TPU batch inference — optimized.
 * Usage: act_infer_batch <cvimodel> <input_dir> <output_dir>
 *
 * Optimizations over the original:
 *  - Preloads ALL inputs into contiguous memory before the inference loop
 *    (eliminates SD card fread per iteration)
 *  - Output buffer allocated once outside the loop
 *    (eliminates per-iteration malloc/free)
 *
 * input_dir contains:
 *   images_000.bin .. images_NNN.bin  — raw FP32 (1,3,224,224)
 *   state_000.bin  .. state_NNN.bin   — raw FP32 (2)
 *   count.txt                        — "N" on first line
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cviruntime.h>

static void fp32_to_bf16(const float *src, uint16_t *dst, int count) {
    for (int i = 0; i < count; i++)
        dst[i] = ((const uint16_t *)src)[i * 2 + 1];
}

static void bf16_to_fp32(const uint16_t *src, float *dst, int count) {
    for (int i = 0; i < count; i++) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        dst[i] = *(const float *)&bits;
    }
}

static int read_raw(const char *path, void *buf, size_t expected) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return -1; }
    size_t n = fread(buf, 1, expected, f);
    fclose(f);
    if (n != expected) { fprintf(stderr, "read %s: got %zu want %zu\n", path, n, expected); return -1; }
    return 0;
}

/*
 * Load all input files into contiguous arrays before the inference loop.
 * Returns 0 on success, -1 on any failure.
 */
static int preload_inputs(const char *in_dir, int num_cases,
                          int img_count, int state_count,
                          float **out_images, float **out_states)
{
    *out_images = (float *)malloc((size_t)num_cases * img_count * sizeof(float));
    *out_states  = (float *)malloc((size_t)num_cases * state_count * sizeof(float));
    if (!*out_images || !*out_states) {
        fprintf(stderr, "preload: malloc failed\n");
        free(*out_images); free(*out_states);
        *out_images = NULL; *out_states = NULL;
        return -1;
    }

    char path[512];
    for (int c = 0; c < num_cases; c++) {
        snprintf(path, sizeof(path), "%s/images_%03d.bin", in_dir, c);
        float *img_buf = *out_images + (size_t)c * img_count;
        if (read_raw(path, img_buf, (size_t)img_count * sizeof(float)) != 0)
            goto fail;
        snprintf(path, sizeof(path), "%s/state_%03d.bin", in_dir, c);
        float *state_buf = *out_states + (size_t)c * state_count;
        if (read_raw(path, state_buf, (size_t)state_count * sizeof(float)) != 0)
            goto fail;
    }
    return 0;

fail:
    free(*out_images); *out_images = NULL;
    free(*out_states); *out_states = NULL;
    return -1;
}

int main(int argc, char *argv[]) {
    const char *model_path = "act_model.cvimodel";
    const char *in_dir = "test_inputs";
    const char *out_dir = "test_outputs";

    if (argc > 1) model_path = argv[1];
    if (argc > 2) in_dir = argv[2];
    if (argc > 3) out_dir = argv[3];

    /* Read test count */
    char path[512];
    snprintf(path, sizeof(path), "%s/count.txt", in_dir);
    FILE *cf = fopen(path, "r");
    if (!cf) { fprintf(stderr, "missing %s\n", path); return 1; }
    int num_cases = 0;
    fscanf(cf, "%d", &num_cases);
    fclose(cf);
    printf("Batch: %d test cases\n", num_cases);

    mkdir(out_dir, 0755);

    /* Load model */
    CVI_MODEL_HANDLE model = NULL;
    CVI_RC ret = CVI_NN_RegisterModel(model_path, &model);
    if (ret != CVI_RC_SUCCESS) { fprintf(stderr, "register model: %d\n", ret); return 1; }

    CVI_TENSOR *inputs, *outputs;
    int32_t in_num, out_num;
    CVI_NN_GetInputOutputTensors(model, &inputs, &in_num, &outputs, &out_num);

    /* Pre-allocate buffers */
    int img_count = 3 * 224 * 224; /* 150528 */
    int state_count = 2;

    float *all_images = NULL;
    float *all_states  = NULL;
    if (preload_inputs(in_dir, num_cases, img_count, state_count,
                       &all_images, &all_states) != 0)
        return 1;

    /* Output buffer — allocated once outside the loop */
    size_t out_count = outputs[0].count;
    float *out_fp32 = (float *)malloc(out_count * sizeof(float));
    if (!out_fp32) { fprintf(stderr, "out_fp32 malloc\n"); return 1; }

    double total_ms = 0;

    for (int c = 0; c < num_cases; c++) {
        float *img_fp32   = all_images + (size_t)c * img_count;
        float *state_fp32 = all_states  + (size_t)c * state_count;

        /* Set inputs */
        for (int i = 0; i < in_num; i++) {
            CVI_TENSOR *t = &inputs[i];
            if (t->count == (size_t)img_count) {
                if (t->fmt == CVI_FMT_BF16)
                    fp32_to_bf16(img_fp32, (uint16_t *)CVI_NN_TensorPtr(t), t->count);
                else
                    memcpy(CVI_NN_TensorPtr(t), img_fp32, t->mem_size);
            } else if (t->count == (size_t)state_count) {
                if (t->fmt == CVI_FMT_BF16)
                    fp32_to_bf16(state_fp32, (uint16_t *)CVI_NN_TensorPtr(t), t->count);
                else
                    memcpy(CVI_NN_TensorPtr(t), state_fp32, t->mem_size);
            }
        }

        /* Run */
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ret = CVI_NN_Forward(model, inputs, in_num, outputs, out_num);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        total_ms += ms;

        /* Read output (reuse out_fp32 buffer) */
        if (outputs[0].fmt == CVI_FMT_BF16)
            bf16_to_fp32((uint16_t *)CVI_NN_TensorPtr(&outputs[0]), out_fp32, out_count);
        else
            memcpy(out_fp32, CVI_NN_TensorPtr(&outputs[0]), out_count * sizeof(float));

        /* Single-line compact format: C<n> <ms> <v0> ... <v15> */
        printf("C%d %.1f", c, ms);
        for (size_t j = 0; j < out_count; j++)
            printf(" %.4f", out_fp32[j]);
        printf("\n");
        fflush(stdout);
    }

    fprintf(stderr, "Done. Total: %.2f ms, Avg: %.2f ms\n", total_ms, total_ms / num_cases);

    CVI_NN_CleanupModel(model);
    free(all_images);
    free(all_states);
    free(out_fp32);
    return 0;
}
