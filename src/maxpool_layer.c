#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
  // Saving our input
  // Probably don't change this
  free_matrix(*l.x);
  *l.x = copy_matrix(in);

  int outw = (l.width-1)/l.stride + 1;
  int outh = (l.height-1)/l.stride + 1;
  int size = l.size;
  int stride = l.stride;
  matrix out = make_matrix_garbage(in.rows, outw*outh*l.channels);

  // TODO: 6.1 - iterate over the input and fill in the output with max values
  // printf("in dimensions: %d, %dx%dx%d\n", in.rows, l.width, l.height, l.channels);
  // printf("out dimensions: %d, %dx%dx%d\n", out.rows, outw, outh, l.channels);
  // printf("stride: %d\n", l.stride);
  // printf("kernel size: %d\n", l.size);

  for (int image_num = 0; image_num < in.rows; image_num++) {
    int image_start_in = image_num * in.cols;
    int image_start_out = image_num * out.cols;
    for (int channel = 0; channel < l.channels; channel++) {
      int channel_in = l.width*l.height*channel;
      int channel_out = outw*outh*channel;
      for (int kern_center_row = 0; kern_center_row < outh*stride; kern_center_row += stride) {
        int row_out = kern_center_row*outw/stride;
        for (int kern_center_col = 0; kern_center_col < outw*stride; kern_center_col+=stride) {
          int col_out = kern_center_col/stride;
          float max = FLT_MIN;
          for (int inner_kern_row = -(size-1)/2; inner_kern_row < size/2 + 1; inner_kern_row++) {
            int row_in = l.width*(kern_center_row + inner_kern_row);
              for (int inner_kern_col = -(size-1)/2; inner_kern_col < size/2 + 1; inner_kern_col++) {
                int col_in = kern_center_col + inner_kern_col;

                int in_pixel = image_start_in + channel_in + row_in + col_in;
                if (!(row_in < 0 || col_in < 0 || row_in / in.cols >= l.height || col_in >= l.width) &&
                    in.data[in_pixel] > max) {
                  max = in.data[in_pixel];
                }
              }
          }
          int out_pixel = image_start_out + channel_out + row_out + col_out;
          out.data[out_pixel] = max;
        }
      }
    }
  }

  return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
  matrix in    = *l.x;
  matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

  int outw = (l.width-1)/l.stride + 1;
  int outh = (l.height-1)/l.stride + 1;
  int size = l.size;
  int stride = l.stride;

  // printf("in dimensions: %d, %dx%dx%d\n", in.rows, l.width, l.height, l.channels);
  // printf("dx dimensions: %d, %dx%dx%d\n", dx.rows, l.width, l.height, l.channels);
  // printf("dy dimensions: %d, %dx%dx%d\n", dy.rows, outw, outh, l.channels);
  // printf("stride: %d\n", l.stride);
  // printf("kernel size: %d\n", l.size);

  // TODO: 6.2 - find the max values in the input again and fill in the
  // corresponding delta with the delta from the output. This should be
  // similar to the forward method in structure.
  for (int image_num = 0; image_num < in.rows; image_num++) {
    int image_start_in = in.cols*image_num;
    int image_start_dy = dy.cols*image_num;
    for (int channel = 0; channel < l.channels; channel++) {
      int channel_in = l.width*l.height*channel;
      int channel_dy = outw*outh*channel;
      for (int kern_center_row = 0; kern_center_row < outh*stride; kern_center_row += stride) {
        int row_out = kern_center_row*outw/stride;
        for (int kern_center_col = 0; kern_center_col < outw*stride; kern_center_col+=stride) {
          int col_out = kern_center_col/stride;
          float max = -FLT_MAX;
          // printf("%f\n", max);
          int max_row = -(size-1)/2 - 1;
          int max_col = -(size-1)/2 - 1;
          for (int inner_kern_row = -(size-1)/2; inner_kern_row < size/2 + 1; inner_kern_row++) {
            int row_in = l.width*(kern_center_row + inner_kern_row);
            for (int inner_kern_col = -(size-1)/2; inner_kern_col < size/2 + 1; inner_kern_col++) {
              int col_in = kern_center_col + inner_kern_col;

              int in_pixel = image_start_in + channel_in + row_in + col_in;

              // printf("%d %d %d %d: %d \n", image_start_in , channel_in , row_in , col_in, in_pixel);
              if (!(row_in < 0 || col_in < 0 || row_in / l.width >= l.height || col_in >= l.width)) {
                if (in.data[in_pixel] > max) {
                  // printf("WE out here2\n");
                  max = in.data[in_pixel];
                  max_row = inner_kern_row;
                  max_col = inner_kern_col;
                }
              }
            }
          }
          assert (max_row != -(size-1)/2 - 1);
          assert (max_col != -(size-1)/2 - 1);

          int dy_pixel = image_start_dy + channel_dy + row_out + col_out;
          int dx_pixel = image_start_in + channel_in + l.width*(kern_center_row + max_row) + kern_center_col + max_col;
          dx.data[dx_pixel] += dy.data[dy_pixel];
        }
      }
    }
  }
  return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}
