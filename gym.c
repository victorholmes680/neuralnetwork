#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <float.h>
#include "raylib.h"
#define SV_IMPLEMENTATION
#include "sv.h"
#define nn_implementation
#include "nn.h"

typedef int Errno;

#define IMG_FACTOR 80
#define IMG_WIDTH (16*IMG_FACTOR)
#define IMG_HEIGHT (9*IMG_FACTOR)

typedef struct {
    size_t *items;
    size_t count;
    size_t capacity;
} Arch;

typedef struct {
    float *items;
    size_t count;
    size_t capacity;
} Cost_Plot;

#define DA_INIT_CAP 256

//  the character \ at the tail of each line is to remove the influence of the '\n' cause the define macro should not contain '\n'
#define da_append(da, item)								\
do {											\
    if(((da))->count >= ((da))->capacity) {						\
	(da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity*2;		\
	(da)->items = realloc((da)->items, (da)->capacity*sizeof(*(da)->items));	\
	assert((da)->items != NULL && "Buy more RAM lol");				\
    }											\
    (da)->items[(da)->count++] = (item);						\
} while(0)										\

char *args_shift(int *argc, char ***argv)
{
    assert(*argc > 0);
    char *result = **argv;
    (*argc) -= 1;
    (*argv) += 1;
    return result;
}


// one neuron network is one plot
// parameters rx and ry mark the start position of the region
// parameters rw and rh limit the width and height of the region
// parameter nn provide the neural network
void nn_render_raylib(NN nn, int rx, int ry, int rw, int rh)
{
    Color low_color = {0xFF, 0x00, 0xFF, 0xFF};
    Color high_color = {0x00, 0xFF, 0x00，0xFF};

    float neuron_radius = rh*0.04;
    int layer_border_vpad = 50; // horizon pad
    int layer_border_hpad = 50; // vertical pad 

    int nn_width = rw - 2*layer_border_hpad; // minus left and right pad
    int nn_height = rh - 2*layer_border_vpad; // minus top and bottom pad

    int nn_x = rx + rw/2 - nn_width/2; // rx + layer_border_hpad
    int nn_y = ry + rh/2 - nn_height/2; // ry + layer_border_vpad

    size_t arch_count = nn.count + 1;
    int layer_hpad = nn_width / arch_count; // the average width of each layer could occupy

    for(size_t l = 0; l < arch_count; ++l) {
	int layer_vpad1 = nn_height / nn.as[l].cols; // the average height of each neural plot of each layer
	for(size_t i = 0; i < nn.as[l].cols; ++i) {
	    int cx1 = nn_x + l*layer_hpad + layer_hpad/2;
	    int cy1 = nn_y + i*layer_vpad1 + layer_vpad1/2;
	    if(l+1 < arch_count) { // draw line to next neural plot if not the last layer
		int layer_vpad2 = nn_height / nn.as[l+1].cols; 
		for(size_t j = 0; j < nn.as[l+1].cols; ++j) {
		    int cx2 = nn_x + (l+1)*layer_hpad + layer_hpad/2;
		    int cy2 = nn_y + j*layer_vpad2 + layer_vpad2/2;
		    float value = sigmoidf(MAT_AT(nn.ws[l], j, i));
		    high_color.a = floorf(255.f*value);
		    float thick = rh*0.004f;
		    Vector2 start = {cx1, cy1};
		    Vector2 end = {cx2, cy2};
		    DrawLineEx(start, end, thick, ColorAlphaBlend(low_color, high_color, WHITE)); // core function which do the operation
		}
	    }

	    if(l > 0) {
		high_color.a = floorf(255.f*sigmoidf(MAT_AT(nn.bs[l-1], 0, i)));
		DrawCircle(cx1, cy1, neuron_radius, ColorAlphaBlend(low_color, high_color, WHITE));
	    } else {
		DrawCircle(cx1, cy1, neuron_radius, GRAY);
	    }
	}
    }
}


void cost_plot_minmax(Cost_Plot plot, float *min, float *max)
{
    *min = FLT_MAX;
    *max = FLT_MIN;
    for(size_t i = 0; i < plot.count; ++i) {
	if(*max < plot.items[i]) *max = plot.items[i];
	if(*min > plot.items[i]) *min = plot.items[i];
    }
}


void plot_cost(Cost_Plot plot, int rx, int ry, int rw, int rh)
{
    float min, max;
    cost_plot_minmax(plot, &min, &max);

    if(min > 0) min = 0;

    size_t n = plot.count;

    if(n < 1000) n = 1000;

    for(size_t i =0; i+1 < plot.count; ++i) {
	float x1 = rx + (float)rw/n*i;
	float y1 = ry + (1 - (plot.items[i] - min)/(max-min))*rh;
	float x2 = rx + (float)rw/n*(i+1);
	float y2 = ry + (1 - (plot.items[i+1] - min)/(max-min))*rh;
	DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, rh*0.005, RED);
    }
}




int main()
{
    return 0;
}
