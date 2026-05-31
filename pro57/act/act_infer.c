/*
 * ACT model TPU inference for SG2002 (CV181x BF16).
 * Loads a cvimodel, runs inference, prints output actions.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <cviruntime.h>

static void fp32_to_bf16(const float *src, uint16_t *dst, int count) {
    for (int i = 0; i < count; i++) {
        /* BF16 = upper 16 bits of FP32 */
        dst[i] = ((const uint16_t *)src)[i * 2 + 1];
    }
}

static void bf16_to_fp32(const uint16_t *src, float *dst, int count) {
    for (int i = 0; i < count; i++) {
        ((uint16_t *)dst)[i * 2 + 1] = src[i];
        ((uint16_t *)dst)[i * 2] = 0;
    }
}

int main(int argc, char *argv[]) {
    const char *model_path = "act_model_det_cv181x_bf16.cvimodel";

    if (argc > 1)
        model_path = argv[1];

    printf("ACT TPU Inference\n");
    printf("Model: %s\n", model_path);

    /* Load model */
    CVI_MODEL_HANDLE model = NULL;
    CVI_RC ret = CVI_NN_RegisterModel(model_path, &model);
    if (ret != CVI_RC_SUCCESS) {
        fprintf(stderr, "Failed to register model: %d\n", ret);
        return 1;
    }

    /* Get input/output tensors */
    CVI_TENSOR *input_tensors, *output_tensors;
    int32_t input_num, output_num;
    ret = CVI_NN_GetInputOutputTensors(model, &input_tensors, &input_num,
                                        &output_tensors, &output_num);
    if (ret != CVI_RC_SUCCESS) {
        fprintf(stderr, "Failed to get tensors: %d\n", ret);
        CVI_NN_CleanupModel(model);
        return 1;
    }

    printf("Inputs: %d\n", input_num);
    for (int i = 0; i < input_num; i++) {
        printf("  [%d] %s shape=(%d,%d,%d,%d) fmt=%d count=%zu\n",
               i, input_tensors[i].name,
               input_tensors[i].shape.dim[0], input_tensors[i].shape.dim[1],
               input_tensors[i].shape.dim[2], input_tensors[i].shape.dim[3],
               input_tensors[i].fmt, input_tensors[i].count);
    }
    printf("Outputs: %d\n", output_num);
    for (int i = 0; i < output_num; i++) {
        printf("  [%d] %s shape=(%d,%d,%d,%d) fmt=%d count=%zu\n",
               i, output_tensors[i].name,
               output_tensors[i].shape.dim[0], output_tensors[i].shape.dim[1],
               output_tensors[i].shape.dim[2], output_tensors[i].shape.dim[3],
               output_tensors[i].fmt, output_tensors[i].count);
    }

    /* Prepare test input (same as x86 simulation) */
    /* images: (1,3,224,224) — use zeros for simplicity (real app would load from camera) */
    /* state: (1,2) = [0.5, -0.3] */
    printf("\nSetting inputs (images=zeros, state=[0.5, -0.3])...\n");

    for (int i = 0; i < input_num; i++) {
        CVI_TENSOR *t = &input_tensors[i];
        if (t->fmt == CVI_FMT_BF16) {
            /* Generate FP32 data then convert to BF16 */
            float *fp32_buf = (float *)calloc(t->count, sizeof(float));
            if (!fp32_buf) {
                fprintf(stderr, "Failed to allocate input buffer\n");
                CVI_NN_CleanupModel(model);
                return 1;
            }

            if (strstr(t->name, "state") || t->count == 2) {
                fp32_buf[0] = 0.5f;
                fp32_buf[1] = -0.3f;
                printf("  state = [%.1f, %.1f]\n", fp32_buf[0], fp32_buf[1]);
            } else {
                /* images — all zeros (baseline test) */
                printf("  images = zeros(%zu)\n", t->count);
            }

            fp32_to_bf16(fp32_buf, (uint16_t *)CVI_NN_TensorPtr(t), t->count);
            free(fp32_buf);
        } else if (t->fmt == CVI_FMT_FP32) {
            memset(CVI_NN_TensorPtr(t), 0, t->mem_size);
            if (strstr(t->name, "state") || t->count == 2) {
                float *buf = (float *)CVI_NN_TensorPtr(t);
                buf[0] = 0.5f;
                buf[1] = -0.3f;
            }
        }
    }

    /* Run inference */
    printf("\nRunning TPU inference...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    ret = CVI_NN_Forward(model, input_tensors, input_num,
                          output_tensors, output_num);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                        (t1.tv_nsec - t0.tv_nsec) / 1e6;

    if (ret != CVI_RC_SUCCESS) {
        fprintf(stderr, "Forward failed: %d\n", ret);
        CVI_NN_CleanupModel(model);
        return 1;
    }
    printf("Inference time: %.2f ms\n", elapsed_ms);

    /* Print output */
    printf("\nOutputs:\n");
    for (int i = 0; i < output_num; i++) {
        CVI_TENSOR *t = &output_tensors[i];
        printf("  [%d] %s shape=(%d,%d,%d,%d)\n",
               i, t->name,
               t->shape.dim[0], t->shape.dim[1],
               t->shape.dim[2], t->shape.dim[3]);

        int print_count = t->count < 16 ? (int)t->count : 16;
        if (t->fmt == CVI_FMT_BF16) {
            float *fp32_buf = (float *)calloc(t->count, sizeof(float));
            bf16_to_fp32((const uint16_t *)CVI_NN_TensorPtr(t), fp32_buf, t->count);
            printf("  values (BF16->FP32): ");
            for (int j = 0; j < print_count; j++) {
                printf("%.6f ", fp32_buf[j]);
            }
            printf("\n");
            free(fp32_buf);
        } else if (t->fmt == CVI_FMT_FP32) {
            float *buf = (float *)CVI_NN_TensorPtr(t);
            printf("  values (FP32): ");
            for (int j = 0; j < print_count; j++) {
                printf("%.6f ", buf[j]);
            }
            printf("\n");
        }
    }

    /* Save output for comparison */
    if (output_num > 0) {
        CVI_TENSOR *t = &output_tensors[0];
        FILE *f = fopen("act_tpu_output.bin", "wb");
        if (f) {
            if (t->fmt == CVI_FMT_BF16) {
                float *fp32_buf = (float *)calloc(t->count, sizeof(float));
                bf16_to_fp32((const uint16_t *)CVI_NN_TensorPtr(t), fp32_buf, t->count);
                fwrite(fp32_buf, sizeof(float), t->count, f);
                free(fp32_buf);
            } else {
                fwrite(CVI_NN_TensorPtr(t), 1, t->mem_size, f);
            }
            fclose(f);
            printf("\nOutput saved to act_tpu_output.bin\n");
        }
    }

    CVI_NN_CleanupModel(model);
    return 0;
}
