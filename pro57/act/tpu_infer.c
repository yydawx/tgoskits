/*
 * ACT model TPU inference — one frame at a time.
 * Usage: tpu_infer <cvimodel> <input_dir>
 *
 * Loads and infers images_000.bin .. images_NNN.bin one by one.
 * Outputs denormalized action values to stdout.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/stat.h>
#include <cviruntime.h>

static const float A_Q01[3] = {-0.1000f, 0.0000f, 0.0000f};
static const float A_Q99[3] = { 0.2000f, 0.2000f, 0.0000f};

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

static inline void denorm(const float *norm, float *raw) {
    for (int d = 0; d < 3; d++) {
        float denom = A_Q99[d] - A_Q01[d];
        if (denom < 0.0001f && denom > -0.0001f)
            raw[d] = 0.0f;
        else
            raw[d] = (norm[d] + 1.0f) / 2.0f * denom + A_Q01[d];
    }
}

int main(int argc, char *argv[]) {
    const char *model_path = "act_model.cvimodel";
    const char *in_dir = "test_inputs";
    if (argc > 1) model_path = argv[1];
    if (argc > 2) in_dir = argv[2];

    char path[512];
    snprintf(path, sizeof(path), "%s/count.txt", in_dir);
    FILE *cf = fopen(path, "r");
    if (!cf) { fprintf(stderr, "missing %s\n", path); return 1; }
    int N = 0; fscanf(cf, "%d", &N); fclose(cf);

    /* Load model */
    CVI_MODEL_HANDLE model = NULL;
    if (CVI_NN_RegisterModel(model_path, &model) != CVI_RC_SUCCESS) {
        fprintf(stderr, "register model failed\n"); return 1;
    }

    CVI_TENSOR *inputs, *outputs;
    int32_t in_num, out_num;
    CVI_NN_GetInputOutputTensors(model, &inputs, &in_num, &outputs, &out_num);

    int img_sz = 3 * 224 * 224;
    int state_sz = 2;
    size_t out_sz = outputs[0].count;
    float *img = (float *)malloc(img_sz * sizeof(float));
    float *state = (float *)malloc(state_sz * sizeof(float));
    float *out = (float *)malloc(out_sz * sizeof(float));
    if (!img || !state || !out) { fprintf(stderr, "malloc\n"); return 1; }

    double total_ms = 0;
    for (int c = 0; c < N; c++) {
        snprintf(path, sizeof(path), "%s/images_%03d.bin", in_dir, c);
        FILE *f = fopen(path, "rb");
        if (!f || fread(img, sizeof(float), img_sz, f) != (size_t)img_sz) {
            fprintf(stderr, "read %s\n", path); return 1;
        }
        fclose(f);

        snprintf(path, sizeof(path), "%s/state_%03d.bin", in_dir, c);
        f = fopen(path, "rb");
        if (!f || fread(state, sizeof(float), state_sz, f) != (size_t)state_sz) {
            fprintf(stderr, "read %s\n", path); return 1;
        }
        fclose(f);

        for (int i = 0; i < in_num; i++) {
            CVI_TENSOR *t = &inputs[i];
            if (t->count == (size_t)img_sz) {
                if (t->fmt == CVI_FMT_BF16)
                    fp32_to_bf16(img, (uint16_t *)CVI_NN_TensorPtr(t), t->count);
                else memcpy(CVI_NN_TensorPtr(t), img, t->mem_size);
            } else if (t->count == (size_t)state_sz) {
                if (t->fmt == CVI_FMT_BF16)
                    fp32_to_bf16(state, (uint16_t *)CVI_NN_TensorPtr(t), t->count);
                else memcpy(CVI_NN_TensorPtr(t), state, t->mem_size);
            }
        }

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CVI_NN_Forward(model, inputs, in_num, outputs, out_num);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)/1e6;
        total_ms += ms;

        if (outputs[0].fmt == CVI_FMT_BF16)
            bf16_to_fp32((uint16_t *)CVI_NN_TensorPtr(&outputs[0]), out, out_sz);
        else memcpy(out, CVI_NN_TensorPtr(&outputs[0]), out_sz * sizeof(float));

        float raw[3];
        denorm(out, raw);
        printf("%d %.1f %.4f %.4f %.4f\n", c, ms, raw[0], raw[1], raw[2]);
        fflush(stdout);
    }

    fprintf(stderr, "Done: %d frames, %.0f ms total, %.1f ms avg\n",
            N, total_ms, total_ms / N);
    CVI_NN_CleanupModel(model);
    free(img); free(state); free(out);
    return 0;
}
