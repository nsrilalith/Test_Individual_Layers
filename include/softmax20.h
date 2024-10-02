#include <stdint.h>

#define SOFTMAX_20_NUM_ROWS 1
#define SOFTMAX_20_ROW_SIZE 20
#define SOFTMAX_20_INPUT_MULT 1077952640
#define SOFTMAX_20_INPUT_LEFT_SHIFT 19
#define SOFTMAX_20_DIFF_MIN -3968
#define SOFTMAX_20_DST_SIZE 20

const int8_t softmax_20_input[20] = {101, 49, 6, -34, -75, -79, -38, 120, -55, 115, 101, 49, 6, -34, -75, -79, -38, 120, -55, 115};

const int8_t softmax_20_output_ref[20] = {-111, -114, -116, -118, -119, -119, -118, -109, -119, -110, -111, -114, -116, -118, -119, -119, -118, -109, -119, -110,};
