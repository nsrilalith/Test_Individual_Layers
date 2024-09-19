#include <stdint.h>

#define SOFTMAX_NUM_ROWS 2
#define SOFTMAX_ROW_SIZE 5
#define SOFTMAX_INPUT_MULT 1077952640
#define SOFTMAX_INPUT_LEFT_SHIFT 19
#define SOFTMAX_DIFF_MIN -3968
#define SOFTMAX_DST_SIZE 10

const int8_t softmax_input[10] = {101, 49, 6, -34, -75, -79, -38, 120, -55, 115};

const int8_t softmax_output_ref[10] = {-57, -70, -79, -86, -92, -94, -88, -54, -91, -56};