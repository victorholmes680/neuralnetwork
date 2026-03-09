#define NN_IMPLEMENTATION
#define NN_ENABLE_GYM
#include "nn.h"

#define BITS 4

size_t arch[] = { 2*BITS, 4*BITS, BITS + 1 };
size_t max_epoch = 100*1000;
size_t epochs_per_frame = 103;
float rate = 1.0f;
bool paused = true;

void verify_nn_adder(Font font, NN nn, float rx, float ry, float rw, float rh)
{
    float s;
    if(rw < rh) {
	s = rw - rw*0.05;
	ry = ry + rh/2 - s/2;
    } else {
	s = rh - rh*0.05;
	rx = rx + rw/2 - s/2;
    }

    size_t n = 1 << BITS;
    float cs = s/n;

    
}
