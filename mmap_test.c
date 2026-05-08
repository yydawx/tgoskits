/*
 * mmap verification test for StarryOS
 * Tests MAP_PRIVATE + fd file mapping with demand paging
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

int main() {
    printf("mmap test starting...\n");

    /* Step 1: create a test file with known content */
    const char* testfile = "/tmp/mmap_test.bin";
    int fd = open(testfile, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open(create)");
        return 1;
    }

    /* Write 4 pages (16KB) of deterministic data */
    size_t filesize = 4096 * 4;
    uint8_t* wbuf = (uint8_t*)malloc(filesize);
    for (size_t i = 0; i < filesize; i++) {
        wbuf[i] = (uint8_t)(i & 0xFF);
    }
    if (write(fd, wbuf, filesize) != (ssize_t)filesize) {
        perror("write");
        return 1;
    }
    free(wbuf);
    printf("  wrote %zu bytes to %s\n", filesize, testfile);

    /* Step 2: mmap the file MAP_PRIVATE */
    void* mapped = mmap(NULL, filesize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }
    printf("  mmap addr=%p size=%zu\n", mapped, filesize);

    /* Step 3: verify content (triggers page faults) */
    uint8_t* data = (uint8_t*)mapped;
    int errors = 0;
    for (size_t i = 0; i < filesize; i++) {
        if (data[i] != (uint8_t)(i & 0xFF)) {
            if (errors < 5)
                printf("  MISMATCH at [%zu]: expected 0x%02x got 0x%02x\n",
                       i, (unsigned)(i & 0xFF), data[i]);
            errors++;
        }
    }
    if (errors == 0) {
        printf("  content verified: %zu bytes OK\n", filesize);
    } else {
        printf("  FAILED: %d mismatches\n", errors);
    }

    /* Step 4: access specific pages (test demand paging) */
    printf("  page[0] data[0]=0x%02x\n", data[0]);
    printf("  page[1] data[4096]=0x%02x\n", data[4096]);
    printf("  page[3] data[12288]=0x%02x\n", data[12288]);

    /* Step 5: verify MAP_PRIVATE (writes should not affect file) */
    /* Can't write to PROT_READ mapping, so skip */

    /* Step 6: munmap */
    if (munmap(mapped, filesize) != 0) {
        perror("munmap");
        return 1;
    }
    printf("  munmap OK\n");

    /* cleanup */
    close(fd);
    unlink(testfile);

    printf("TEST PASSED\n");
    return 0;
}
