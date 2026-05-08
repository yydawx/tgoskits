/*
 * Minimal float diagnostic - verify basic arithmetic on RISC-V
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static void print_hex_f(const char* label, float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    printf("%s: float=%.8f  hex=0x%08x\n", label, f, bits);
}

int main() {
    printf("float diagnostic starting...\n\n");

    /* Test 1: constant float */
    print_hex_f("const -0.8", -0.8f);
    print_hex_f("const  0.1", 0.1f);
    print_hex_f("const  0.0", 0.0f);

    /* Test 2: integer arithmetic in float */
    int j0 = 0;
    int rem0 = j0 % 17;
    int sub0 = rem0 - 8;
    float f0 = (float)sub0 * 0.1f;
    printf("\nj=0: rem=%d  sub=%d\n", rem0, sub0);
    print_hex_f("  (0%17 - 8) * 0.1", f0);

    int j1 = 1;
    int rem1 = j1 % 17;
    int sub1 = rem1 - 8;
    float f1 = (float)sub1 * 0.1f;
    printf("j=1: rem=%d  sub=%d\n", rem1, sub1);
    print_hex_f("  (1%17 - 8) * 0.1", f1);

    /* Test 3: loop - fill array */
    printf("\nArray fill test:\n");
    float arr[8];
    for (int i = 0; i < 8; i++) {
        arr[i] = (float)((i % 17) - 8) * 0.1f;
    }
    for (int i = 0; i < 8; i++) {
        char label[32];
        snprintf(label, sizeof(label), "  arr[%d]", i);
        print_hex_f(label, arr[i]);
    }

    /* Test 4: calloc + fill */
    printf("\ncalloc fill test:\n");
    float* buf = (float*)calloc(10, sizeof(float));
    for (int i = 0; i < 10; i++) {
        buf[i] = (float)((i % 17) - 8) * 0.1f;
    }
    for (int i = 0; i < 8; i++) {
        char label[32];
        snprintf(label, sizeof(label), "  buf[%d]", i);
        print_hex_f(label, buf[i]);
    }
    free(buf);

    printf("\nTEST PASSED\n");
    return 0;
}
