// Intentionally vulnerable C++ sample for audit tests.
#include <cstdlib>
#include <cstring>

void vulnerable(char *src) {
    char *buf = (char *)malloc(128);
    strcpy(buf, src);
    printf(buf);
    free(buf);
    printf("%s", buf);
    char *x = (char *)malloc(64);
}
