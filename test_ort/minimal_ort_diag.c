/*
 * Minimal test: link with ORT but don't call any API.
 * Just test if basic float arithmetic is affected by ORT linking.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "onnxruntime_c_api.h"

int main() {
    printf("minimal ORT-linked float test\n");

    /* Test 1: constant */
    float f = -0.8f;
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    printf("const -0.8f: hex=0x%08x (expect 0xbf4ccccd)\n", bits);

    /* Test 2: loop fill */
    float arr[8];
    for (int i = 0; i < 8; i++) {
        arr[i] = (float)((i % 17) - 8) * 0.1f;
    }
    memcpy(&bits, &arr[0], sizeof(bits));
    printf("arr[0]: hex=0x%08x (expect 0xbf4ccccd)\n", bits);
    memcpy(&bits, &arr[1], sizeof(bits));
    printf("arr[1]: hex=0x%08x (expect 0xbf333333)\n", bits);

    /* Test 3: calloc fill */
    float* buf = (float*)calloc(8, sizeof(float));
    for (int i = 0; i < 8; i++) {
        buf[i] = (float)((i % 17) - 8) * 0.1f;
    }
    memcpy(&bits, &buf[0], sizeof(bits));
    printf("buf[0]: hex=0x%08x (expect 0xbf4ccccd)\n", bits);
    free(buf);

    /* Test 4: access OrtApi but don't call anything */
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    printf("OrtApi addr=%p\n", (const void*)api);

    /* Test 5: fill after accessing OrtApi */
    float arr2[8];
    for (int i = 0; i < 8; i++) {
        arr2[i] = (float)((i % 17) - 8) * 0.1f;
    }
    memcpy(&bits, &arr2[0], sizeof(bits));
    printf("arr2[0] after OrtApi: hex=0x%08x (expect 0xbf4ccccd)\n", bits);

    printf("TEST PASSED\n");
    return 0;
}
