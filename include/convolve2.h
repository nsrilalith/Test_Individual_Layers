// Generated by generate_test_data.py using tensorflow version 2.10.0 (Keras version 2.10.0).
// Interpreter from tensorflow version 2.10.0 and revision upstream/v2.10.0-0-g359c3cdfc5f.
#include <stdint.h>

#define CONV_2_OUT_CH 4
#define CONV_2_IN_CH 2
#define CONV_2_INPUT_W 6
#define CONV_2_INPUT_H 3
#define CONV_2_DST_SIZE 72
#define CONV_2_INPUT_SIZE 36
#define CONV_2_OUT_ACTIVATION_MIN -101
#define CONV_2_OUT_ACTIVATION_MAX 127
#define CONV_2_INPUT_BATCHES 1
#define CONV_2_FILTER_X 3
#define CONV_2_FILTER_Y 3
#define CONV_2_STRIDE_X 1
#define CONV_2_STRIDE_Y 1
#define CONV_2_PAD_X 1
#define CONV_2_PAD_Y 1
#define CONV_2_OUTPUT_W 6
#define CONV_2_OUTPUT_H 3
#define CONV_2_INPUT_OFFSET 128
#define CONV_2_OUTPUT_OFFSET -3
#define CONV_2_DILATION_X 1
#define CONV_2_DILATION_Y 1

const int32_t conv_2_biases[4] = {-12214, -32460, -9046, 34979};

const int8_t conv_2_weights[72] = {0,   103, -58, 49,  -54,  117,  113, 37,   -127, 113,  31,  -67, 52,  -115, 12,
                                   -69, 37,  -53, -85, 105,  42,   -91, 86,   15,   -127, -6,  15,  25,  124,  75,
                                   -31, -16, -63, -21, -126, -116, 62,  -31,  87,   -114, -50, 127, -60, 29,   28,
                                   109, 50,  -63, 99,  8,    77,   7,   92,   -82,  -121, -42, 107, -50, 52,   -73,
                                   -81, 120, 71,  14,  78,   45,   97,  -127, -6,   106,  61,  -95};

const int8_t conv_2_input[36] = {115,  -124, -58,  -108, -57,  -100, 7,   31,  -54, 95,  -8,  -51,
                                 -115, -57,  -81,  -53,  -21,  -127, -59, -97, 126, 112, -13, -108,
                                 108,  -121, -119, -10,  -112, -9,   100, -59, -33, -36, 119, 38};

const int8_t conv_2_output_ref[72] = {
    -64, -47, -3,  67, -2,  -91, -5,  36, -28, -44, 5,  55, -21, -100, 20, 72, 13, -82,  62,  106, -24, -76, 48, 66,
    -38, -44, 29,  66, -13, -74, 50,  63, -32, -75, 40, 57, -9,  -50,  36, 38, 20, -101, 65,  66,  20,  -93, 19, 61,
    -53, -27, -14, 57, 25,  -50, -38, 29, 8,   18,  28, 84, -26, 3,    39, 71, 6,  -26,  -17, 76,  7,   -41, 33, 30};

const int32_t conv_2_output_mult[4] = {1374632181, 1365315267, 1377078717, 1252339957};

const int32_t conv_2_output_shift[4] = {-9, -9, -9, -9};
