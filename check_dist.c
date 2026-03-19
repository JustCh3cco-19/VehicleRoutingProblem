#include <stdio.h>
#include <math.h>
int main() {
    double x0=51, y0=92;
    double x1=14, y1=71;
    double d = sqrt(pow(x0-x1, 2) + pow(y0-y1, 2));
    printf("d(0, 1) = %f\n", d);
    return 0;
}
