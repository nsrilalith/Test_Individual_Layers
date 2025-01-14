// Generated by generate_test_data.py using TFL version 2.6.0 as reference.
#include <stdint.h>
#define MAXPOOLING_20_OUT_CH 1
#define MAXPOOLING_20_IN_CH 1
#define MAXPOOLING_20_INPUT_W 10
#define MAXPOOLING_20_INPUT_H 8
#define MAXPOOLING_20_DST_SIZE 20
#define MAXPOOLING_20_INPUT_SIZE 80
#define MAXPOOLING_20_OUT_ACTIVATION_MIN -128
#define MAXPOOLING_20_OUT_ACTIVATION_MAX 127
#define MAXPOOLING_20_INPUT_BATCHES 1
#define MAXPOOLING_20_FILTER_X 2
#define MAXPOOLING_20_FILTER_Y 2
#define MAXPOOLING_20_STRIDE_X 2
#define MAXPOOLING_20_STRIDE_Y 2
#define MAXPOOLING_20_PAD_X 0
#define MAXPOOLING_20_PAD_Y 0
#define MAXPOOLING_20_OUTPUT_W 5
#define MAXPOOLING_20_OUTPUT_H 4

const int8_t maxpooling_20_input[80] = {-117, -127, -44, 5, 13, 5, 26, -115, -33, -102, 91, 45, 68, 52,
                                        60, 93, 96, -73, -29, -46, -128, 62, -108, 20, 67, 84, 109, -67,
                                        -70, -99, -10, -60, -55, -9, 56, -60, -74, -52, -126, 14, -117, 
                                        -127, -44, 5, 13, 5, 26, -115, -33, -102, 91, 45, 68, 52,
                                        60, 93, 96, -73, -29, -46, -128, 62, -108, 20, 67, 84, 109, -67,
                                        -70, -99, -10, -60, -55, -9, 56, -60, -74, -52, -126, 14};

const int8_t maxpooling_20_output_ref[20] = {91, 68, 93, 96, -29, 62, 20, 84, 109, 14, 91, 68, 93, 96, -29, 62, 20, 84, 109, 14};
