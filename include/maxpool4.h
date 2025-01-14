// Generated by generate_test_data.py using TFL version 2.6.0 as reference.
#include <stdint.h>
#define MAXPOOLING_4_OUT_CH 2
#define MAXPOOLING_4_IN_CH 2
#define MAXPOOLING_4_INPUT_W 1
#define MAXPOOLING_4_INPUT_H 20
#define MAXPOOLING_4_DST_SIZE 14
#define MAXPOOLING_4_INPUT_SIZE 40
#define MAXPOOLING_4_OUT_ACTIVATION_MIN -128
#define MAXPOOLING_4_OUT_ACTIVATION_MAX 127
#define MAXPOOLING_4_INPUT_BATCHES 1
#define MAXPOOLING_4_FILTER_X 1
#define MAXPOOLING_4_FILTER_Y 3
#define MAXPOOLING_4_STRIDE_X 1
#define MAXPOOLING_4_STRIDE_Y 3
#define MAXPOOLING_4_PAD_X 0
#define MAXPOOLING_4_PAD_Y 0
#define MAXPOOLING_4_OUTPUT_W 1
#define MAXPOOLING_4_OUTPUT_H 7

const int8_t maxpooling_4_input[40] = {-117, -127, -44, 5,   13,  5,   26,   -115, -33,  -102, 91,   45, 68,  52,
                                       60,   93,   96,  -73, -29, -46, -128, 62,   -108, 20,   67,   84, 109, -67,
                                       -70,  -99,  -10, -60, -55, -9,  56,   -60,  -74,  -52,  -126, 14};

const int8_t maxpooling_4_output_ref[14] = {13, 5, 91, 45, 96, 93, -29, 62, 109, 84, 56, -9, -74, 14};
