#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include "net.h"

static void print_heap(const char* label) {
    void* brk = sbrk(0);
    printf("[MEM-ncnn] %s brk=%p\n", label, brk);
}

static void print_float_hex(const char* label, const float* data, int n) {
    printf("%s:", label);
    for (int i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &data[i], sizeof(bits));
        printf(" 0x%08x", bits);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    const char* param_path = "/model.ncnn.param";
    const char* bin_path = "/model.ncnn.bin";
    if (argc > 2) {
        param_path = argv[1];
        bin_path = argv[2];
    }

    printf("ncnn inference test starting...\n");
    print_heap("start");

    ncnn::Net net;
    net.opt.num_threads = 1;
    net.opt.use_vulkan_compute = false;
    net.opt.lightmode = true;  /* free intermediate blobs early */

    printf("Loading model: %s / %s\n", param_path, bin_path);
    print_heap("before model load");

    int ret = net.load_param(param_path);
    if (ret != 0) {
        fprintf(stderr, "Error: load_param failed %d\n", ret);
        return 1;
    }
    ret = net.load_model(bin_path);
    if (ret != 0) {
        fprintf(stderr, "Error: load_model failed %d\n", ret);
        return 1;
    }
    print_heap("after model load");
    printf("Model loaded\n");

    /* Create inputs same as ort_test.c pattern */
    ncnn::Mat in0(224, 224, 3);  /* image: [H, W, C] */
    ncnn::Mat in1(2);             /* state: [state_dim] */

    float* img_data = (float*)in0.data;
    size_t img_n = 224 * 224 * 3;
    printf("Filling image input (%zu elements)\n", img_n);
    for (size_t j = 0; j < img_n; j++) {
        int rem = (int)(j % 17);
        int sub = rem - 8;
        img_data[j] = (float)sub * 0.1f;
    }

    float* state_data = (float*)in1.data;
    for (int i = 0; i < 2; i++) {
        int rem = i % 17;
        int sub = rem - 8;
        state_data[i] = (float)sub * 0.1f;
    }

    print_float_hex("  in0 first 4", img_data, 4);
    print_float_hex("  in1 first 2", state_data, 2);

    /* Run inference */
    printf("Running inference...\n");
    fflush(stdout);
    print_heap("before inference");

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in0);
    ex.input("in1", in1);

    ncnn::Mat out;
    ret = ex.extract("out0", out);
    if (ret != 0) {
        fprintf(stderr, "Error: extract failed %d\n", ret);
        return 1;
    }

    print_heap("after inference");
    printf("Inference done\n");

    printf("Output shape: [%d, %d, %d]\n", out.w, out.h, out.c);
    size_t on = out.total();
    printf("Output size: %zu elements\n", on);

    float* out_data = (float*)out.data;
    printf("output (%zu values):\n", on);
    for (size_t i = 0; i < on && i < 256; i++) {
        uint32_t bits;
        memcpy(&bits, &out_data[i], sizeof(bits));
        printf("  [%3zu] 0x%08x\n", i, bits);
    }

    printf("TEST PASSED\n");
    return 0;
}
