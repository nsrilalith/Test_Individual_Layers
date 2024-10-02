#include "arm_nnfunctions.h"
#include "maxpool.h"
#include "maxpool2.h"
#include "maxpool4.h"
#include "maxpool5.h"
#include "convolve.h"
#include "convolve2.h"
#include "convolve3.h"
#include "convolve4.h"
#include "convolve5.h"
#include "convolve6.h"
#include "softmax.h"
#include "softmax20.h"
#include <stdio.h>

void maxpooling_arm_max_pool_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[MAXPOOLING_DST_SIZE] = {0};

    int flag;

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = maxpooling_input;

    input_dims.n = MAXPOOLING_INPUT_BATCHES;
    input_dims.w = MAXPOOLING_INPUT_W;
    input_dims.h = MAXPOOLING_INPUT_H;
    input_dims.c = MAXPOOLING_IN_CH;
    filter_dims.w = MAXPOOLING_FILTER_X;
    filter_dims.h = MAXPOOLING_FILTER_Y;
    output_dims.w = MAXPOOLING_OUTPUT_W;
    output_dims.h = MAXPOOLING_OUTPUT_H;
    output_dims.c = MAXPOOLING_OUT_CH;

    pool_params.padding.w = MAXPOOLING_PAD_X;
    pool_params.padding.h = MAXPOOLING_PAD_Y;
    pool_params.stride.w = MAXPOOLING_STRIDE_X;
    pool_params.stride.h = MAXPOOLING_STRIDE_Y;

    pool_params.activation.min = MAXPOOLING_OUT_ACTIVATION_MIN;
    pool_params.activation.max = MAXPOOLING_OUT_ACTIVATION_MAX;

    
    arm_cmsis_nn_status result = arm_max_pool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

    for (int i = 0; i<MAXPOOLING_DST_SIZE; i++){
        if (output[i] != maxpooling_output_ref[i]){
            flag = 0;
        }
        else{
            flag = 1;
        }
    }
    if(flag == 1){
        printf("Max Pooling Test Passed\n");
    }

    printf("Output Matrix: \n");
    for (int i = 0; i<MAXPOOLING_DST_SIZE; i++){
        printf("%d ", output[i]);
    }

    printf("\n");
    printf("Output Expected Matrix: \n");
    for (int i = 0; i<MAXPOOLING_DST_SIZE; i++){
        printf("%d ", maxpooling_output_ref[i]);
    }
    printf("\n");
}
void maxpooling_2_arm_max_pool_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[MAXPOOLING_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = maxpooling_2_input;

    input_dims.n = MAXPOOLING_2_INPUT_BATCHES;
    input_dims.w = MAXPOOLING_2_INPUT_W;
    input_dims.h = MAXPOOLING_2_INPUT_H;
    input_dims.c = MAXPOOLING_2_IN_CH;
    filter_dims.w = MAXPOOLING_2_FILTER_X;
    filter_dims.h = MAXPOOLING_2_FILTER_Y;
    output_dims.w = MAXPOOLING_2_OUTPUT_W;
    output_dims.h = MAXPOOLING_2_OUTPUT_H;
    output_dims.c = MAXPOOLING_2_OUT_CH;

    pool_params.padding.w = MAXPOOLING_2_PAD_X;
    pool_params.padding.h = MAXPOOLING_2_PAD_Y;
    pool_params.stride.w = MAXPOOLING_2_STRIDE_X;
    pool_params.stride.h = MAXPOOLING_2_STRIDE_Y;

    pool_params.activation.min = MAXPOOLING_2_OUT_ACTIVATION_MIN;
    pool_params.activation.max = MAXPOOLING_2_OUT_ACTIVATION_MAX;

    
    
    arm_cmsis_nn_status result = arm_max_pool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

    int flag;
    for (int i = 0; i<MAXPOOLING_2_DST_SIZE; i++){
        if (output[i] != maxpooling_2_output_ref[i]){
            flag = 0;
        }
        else{
            flag = 1;
        }
    }
    if(flag == 1){
        printf("Max Pooling Test Passed\n");
    }

    printf("Output Matrix: \n");
    for (int i = 0; i<MAXPOOLING_2_DST_SIZE; i++){
        printf("%d ", output[i]);
    }

    printf("\n");
    printf("Output Expected Matrix: \n");
    for (int i = 0; i<MAXPOOLING_2_DST_SIZE; i++){
        printf("%d ", maxpooling_2_output_ref[i]);
    }
    printf("\n");
}

void maxpooling_4_arm_max_pool_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[MAXPOOLING_4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = maxpooling_4_input;

    input_dims.n = MAXPOOLING_4_INPUT_BATCHES;
    input_dims.w = MAXPOOLING_4_INPUT_W;
    input_dims.h = MAXPOOLING_4_INPUT_H;
    input_dims.c = MAXPOOLING_4_IN_CH;
    filter_dims.w = MAXPOOLING_4_FILTER_X;
    filter_dims.h = MAXPOOLING_4_FILTER_Y;
    output_dims.w = MAXPOOLING_4_OUTPUT_W;
    output_dims.h = MAXPOOLING_4_OUTPUT_H;
    output_dims.c = MAXPOOLING_4_OUT_CH;

    pool_params.padding.w = MAXPOOLING_4_PAD_X;
    pool_params.padding.h = MAXPOOLING_4_PAD_Y;
    pool_params.stride.w = MAXPOOLING_4_STRIDE_X;
    pool_params.stride.h = MAXPOOLING_4_STRIDE_Y;

    pool_params.activation.min = MAXPOOLING_4_OUT_ACTIVATION_MIN;
    pool_params.activation.max = MAXPOOLING_4_OUT_ACTIVATION_MAX;

   
    arm_cmsis_nn_status result = arm_max_pool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

    int flag;
    for (int i = 0; i<MAXPOOLING_4_DST_SIZE; i++){
        if (output[i] != maxpooling_4_output_ref[i]){
            flag = 0;
        }
        else{
            flag = 1;
        }
    }
    if(flag == 1){
        printf("Max Pooling Test Passed\n");
    }

    printf("Output Matrix: \n");
    for (int i = 0; i<MAXPOOLING_4_DST_SIZE; i++){
        printf("%d ", output[i]);
    }

    printf("\n");
    printf("Output Expected Matrix: \n");
    for (int i = 0; i<MAXPOOLING_4_DST_SIZE; i++){
        printf("%d ", maxpooling_4_output_ref[i]);
    }
    printf("\n");
        
}

void maxpooling_20_neuron(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[MAXPOOLING_20_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = maxpooling_20_input;

    input_dims.n = MAXPOOLING_20_INPUT_BATCHES;
    input_dims.w = MAXPOOLING_20_INPUT_W;
    input_dims.h = MAXPOOLING_20_INPUT_H;
    input_dims.c = MAXPOOLING_20_IN_CH;
    filter_dims.w = MAXPOOLING_20_FILTER_X;
    filter_dims.h = MAXPOOLING_20_FILTER_Y;
    output_dims.w = MAXPOOLING_20_OUTPUT_W;
    output_dims.h = MAXPOOLING_20_OUTPUT_H;
    output_dims.c = MAXPOOLING_20_OUT_CH;

    pool_params.padding.w = MAXPOOLING_20_PAD_X;
    pool_params.padding.h = MAXPOOLING_20_PAD_Y;
    pool_params.stride.w = MAXPOOLING_20_STRIDE_X;
    pool_params.stride.h = MAXPOOLING_20_STRIDE_Y;

    pool_params.activation.min = MAXPOOLING_20_OUT_ACTIVATION_MIN;
    pool_params.activation.max = MAXPOOLING_20_OUT_ACTIVATION_MAX;

   
    arm_cmsis_nn_status result = arm_max_pool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

    int flag = 0;
    for (int i = 0; i<MAXPOOLING_20_DST_SIZE; i++){
        if (output[i] != maxpooling_20_output_ref[i]){
            flag = 0;
        }
        else{
            flag = 1;
        }
    }
    if(flag == 1){
        printf("Max Pooling Test Passed\n");
    }

    printf("Output Matrix: \n");
    for (int i = 0; i<MAXPOOLING_20_DST_SIZE; i++){
        printf("%d, ", output[i]);
    }

    printf("\n");
    printf("Output Expected Matrix: \n");
    for (int i = 0; i<MAXPOOLING_20_DST_SIZE; i++){
        printf("%d ", maxpooling_20_output_ref[i]);
    }
    printf("\n");
        
}

void basic_2_arm_convolve_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[BASIC_2_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = basic_2_int4_biases;
    const int8_t *kernel_data = basic_2_int4_weights;
    const int8_t *input_data = basic_2_int4_input;
    const int8_t *output_ref = basic_2_int4_output_ref;
    const int32_t output_ref_size = BASIC_2_INT4_DST_SIZE;

    input_dims.n = BASIC_2_INT4_INPUT_BATCHES;
    input_dims.w = BASIC_2_INT4_INPUT_W;
    input_dims.h = BASIC_2_INT4_INPUT_H;
    input_dims.c = BASIC_2_INT4_IN_CH;
    filter_dims.w = BASIC_2_INT4_FILTER_X;
    filter_dims.h = BASIC_2_INT4_FILTER_Y;
    output_dims.w = BASIC_2_INT4_OUTPUT_W;
    output_dims.h = BASIC_2_INT4_OUTPUT_H;
    output_dims.c = BASIC_2_INT4_OUT_CH;

    conv_params.padding.w = BASIC_2_INT4_PAD_X;
    conv_params.padding.h = BASIC_2_INT4_PAD_Y;
    conv_params.stride.w = BASIC_2_INT4_STRIDE_X;
    conv_params.stride.h = BASIC_2_INT4_STRIDE_Y;
    conv_params.dilation.w = BASIC_2_INT4_DILATION_X;
    conv_params.dilation.h = BASIC_2_INT4_DILATION_Y;

    conv_params.input_offset = BASIC_2_INT4_INPUT_OFFSET;
    conv_params.output_offset = BASIC_2_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = BASIC_2_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = BASIC_2_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)basic_2_int4_output_mult;
    quant_params.shift = (int32_t *)basic_2_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s4(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 &output_dims,
                                                 output);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    int flag;
    for (int i = 0; i<BASIC_2_INT4_DST_SIZE; i++){
        if (output[i] != output_ref[i]){
            flag = 0;
        }
        else{
            flag = 1;
        }
    }

    if (flag == 1){
        printf("\nConvolve Passed\n");
    }

    printf("\nComputed Output: \n");
    for (int i = 0; i<BASIC_2_INT4_DST_SIZE; i++){
        printf("%d ", output[i]);
    }
    printf("\n");

    printf("\nActualy Output: \n");
    for (int i = 0; i<BASIC_2_INT4_DST_SIZE; i++){
        printf("%d ", output_ref[i]);
    }
    printf("\n");
}

void stride2pad1_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[STRIDE2PAD1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = stride2pad1_biases;
    const int8_t *kernel_data = stride2pad1_weights;
    const int8_t *input_data = stride2pad1_input;
    const int8_t *output_ref = stride2pad1_output_ref;
    const int32_t output_ref_size = STRIDE2PAD1_DST_SIZE;

    input_dims.n = STRIDE2PAD1_INPUT_BATCHES;
    input_dims.w = STRIDE2PAD1_INPUT_W;
    input_dims.h = STRIDE2PAD1_INPUT_H;
    input_dims.c = STRIDE2PAD1_IN_CH;
    filter_dims.w = STRIDE2PAD1_FILTER_X;
    filter_dims.h = STRIDE2PAD1_FILTER_Y;
    filter_dims.c = STRIDE2PAD1_IN_CH;
    output_dims.w = STRIDE2PAD1_OUTPUT_W;
    output_dims.h = STRIDE2PAD1_OUTPUT_H;
    output_dims.c = STRIDE2PAD1_OUT_CH;

    conv_params.padding.w = STRIDE2PAD1_PAD_X;
    conv_params.padding.h = STRIDE2PAD1_PAD_Y;
    conv_params.stride.w = STRIDE2PAD1_STRIDE_X;
    conv_params.stride.h = STRIDE2PAD1_STRIDE_Y;
    conv_params.dilation.w = STRIDE2PAD1_DILATION_X;
    conv_params.dilation.h = STRIDE2PAD1_DILATION_Y;

    conv_params.input_offset = STRIDE2PAD1_INPUT_OFFSET;
    conv_params.output_offset = STRIDE2PAD1_OUTPUT_OFFSET;
    conv_params.activation.min = STRIDE2PAD1_OUT_ACTIVATION_MIN;
    conv_params.activation.max = STRIDE2PAD1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)stride2pad1_output_mult;
    quant_params.shift = (int32_t *)stride2pad1_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 &output_dims,
                                                 output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    int flag;
    for (int i = 0; i<STRIDE2PAD1_DST_SIZE; i++){
        if (output[i] != output_ref[i]){
            flag = 0;
        }
        else{
            flag = 1;
        }
    }

    if (flag == 1){
        printf("\nConvolve Passed\n");
    }

    printf("\nComputed Output: \n");
    for (int i = 0; i<STRIDE2PAD1_DST_SIZE; i++){
        printf("%d ", output[i]);
    }
    printf("\n");

    printf("\nActualy Output: \n");
    for (int i = 0; i<STRIDE2PAD1_DST_SIZE; i++){
        printf("%d ", output_ref[i]);
    }
    printf("\n");
}

void conv_2_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_2_biases;
    const int8_t *kernel_data = conv_2_weights;
    const int8_t *input_data = conv_2_input;
    const int8_t *output_ref = conv_2_output_ref;
    const int32_t output_ref_size = CONV_2_DST_SIZE;

    input_dims.n = CONV_2_INPUT_BATCHES;
    input_dims.w = CONV_2_INPUT_W;
    input_dims.h = CONV_2_INPUT_H;
    input_dims.c = CONV_2_IN_CH;
    filter_dims.w = CONV_2_FILTER_X;
    filter_dims.h = CONV_2_FILTER_Y;
    filter_dims.c = CONV_2_IN_CH;
    output_dims.w = CONV_2_OUTPUT_W;
    output_dims.h = CONV_2_OUTPUT_H;
    output_dims.c = CONV_2_OUT_CH;

    conv_params.padding.w = CONV_2_PAD_X;
    conv_params.padding.h = CONV_2_PAD_Y;
    conv_params.stride.w = CONV_2_STRIDE_X;
    conv_params.stride.h = CONV_2_STRIDE_Y;
    conv_params.dilation.w = CONV_2_DILATION_X;
    conv_params.dilation.h = CONV_2_DILATION_Y;

    conv_params.input_offset = CONV_2_INPUT_OFFSET;
    conv_params.output_offset = CONV_2_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_2_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_2_output_mult;
    quant_params.shift = (int32_t *)conv_2_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 conv_2_weights,
                                                 &bias_dims,
                                                 bias_data,
                                                 &output_dims,
                                                 output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    int flag;
    for (int i = 0; i<CONV_2_DST_SIZE; i++){
        if (output[i] != output_ref[i]){
            flag = 0;
        }
        else{
            flag = 1;
        }
    }

    if (flag == 1){
        printf("\nConvolve Passed\n");
    }

    printf("\nComputed Output: \n");
    for (int i = 0; i<CONV_2_DST_SIZE; i++){
        printf("%d ", output[i]);
    }
    printf("\n");

    printf("\nActualy Output: \n");
    for (int i = 0; i<CONV_2_DST_SIZE; i++){
        printf("%d ", output_ref[i]);
    }
    printf("\n");
}

void conv_3_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_3_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_3_biases;
    const int8_t *kernel_data = conv_3_weights;
    const int8_t *input_data = conv_3_input;
    const int8_t *output_ref = conv_3_output_ref;
    const int32_t output_ref_size = CONV_3_DST_SIZE;

    input_dims.n = CONV_3_INPUT_BATCHES;
    input_dims.w = CONV_3_INPUT_W;
    input_dims.h = CONV_3_INPUT_H;
    input_dims.c = CONV_3_IN_CH;
    filter_dims.w = CONV_3_FILTER_X;
    filter_dims.h = CONV_3_FILTER_Y;
    filter_dims.c = CONV_3_IN_CH;
    output_dims.w = CONV_3_OUTPUT_W;
    output_dims.h = CONV_3_OUTPUT_H;
    output_dims.c = CONV_3_OUT_CH;

    conv_params.padding.w = CONV_3_PAD_X;
    conv_params.padding.h = CONV_3_PAD_Y;
    conv_params.stride.w = CONV_3_STRIDE_X;
    conv_params.stride.h = CONV_3_STRIDE_Y;
    conv_params.dilation.w = CONV_3_DILATION_X;
    conv_params.dilation.h = CONV_3_DILATION_Y;

    conv_params.input_offset = CONV_3_INPUT_OFFSET;
    conv_params.output_offset = CONV_3_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_3_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_3_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_3_output_mult;
    quant_params.shift = (int32_t *)conv_3_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 conv_3_weights,
                                                 &bias_dims,
                                                 bias_data,
                                                 &output_dims,
                                                 output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    int flag;
    for (int i = 0; i<CONV_3_DST_SIZE; i++){
        if (output[i] != output_ref[i]){
            flag = 0;
        }
        else{
            flag = 1;
        }
    }

    if (flag == 1){
        printf("\nConvolve Passed\n");
    }

    printf("\nComputed Output: \n");
    for (int i = 0; i<CONV_3_DST_SIZE; i++){
        printf("%d ", output[i]);
    }
    printf("\n");

    printf("\nActualy Output: \n");
    for (int i = 0; i<CONV_3_DST_SIZE; i++){
        printf("%d ", output_ref[i]);
    }
    printf("\n");
}

void kernel1x1_arm_convolve_1x1_s8_fast(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[KERNEL1X1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = kernel1x1_biases;
    const int8_t *input_data = kernel1x1_input;
    const int8_t *output_ref = kernel1x1_output_ref;
    const int32_t *output_ref_size = KERNEL1X1_DST_SIZE;

    input_dims.n = KERNEL1X1_INPUT_BATCHES;
    input_dims.h = KERNEL1X1_INPUT_H;
    input_dims.w = KERNEL1X1_INPUT_W;
    input_dims.c = KERNEL1X1_IN_CH;
    filter_dims.n = KERNEL1X1_OUT_CH;
    filter_dims.h = KERNEL1X1_FILTER_Y;
    filter_dims.w = KERNEL1X1_FILTER_X;
    filter_dims.c = KERNEL1X1_IN_CH;
    output_dims.n = KERNEL1X1_INPUT_BATCHES;
    output_dims.h = KERNEL1X1_OUTPUT_H;
    output_dims.w = KERNEL1X1_OUTPUT_W;
    output_dims.c = KERNEL1X1_OUT_CH;

    conv_params.padding.h = KERNEL1X1_PAD_Y;
    conv_params.padding.w = KERNEL1X1_PAD_X;
    conv_params.stride.h = KERNEL1X1_STRIDE_Y;
    conv_params.stride.w = KERNEL1X1_STRIDE_X;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    conv_params.input_offset = KERNEL1X1_INPUT_OFFSET;
    conv_params.output_offset = KERNEL1X1_OUTPUT_OFFSET;
    conv_params.activation.min = KERNEL1X1_OUT_ACTIVATION_MIN;
    conv_params.activation.max = KERNEL1X1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)kernel1x1_output_mult;
    quant_params.shift = (int32_t *)kernel1x1_output_shift;

    const int32_t buf_size = arm_convolve_1x1_s8_fast_get_buffer_size(&input_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_1x1_s8_fast(&ctx,
                                                          &conv_params,
                                                          &quant_params,
                                                          &input_dims,
                                                          input_data,
                                                          &filter_dims,
                                                          kernel1x1_weights,
                                                          &bias_dims,
                                                          bias_data,
                                                          &output_dims,
                                                          output);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    int flag;
    for (int i = 0; i<KERNEL1X1_DST_SIZE; i++){
        if (output[i] != output_ref[i]){
            flag = 0;
        }
        else{
            flag = 1;
        }
    }

    if (flag == 1){
        printf("\nConvolve Passed\n");
    }

    printf("\nComputed Output: \n");
    for (int i = 0; i<KERNEL1X1_DST_SIZE; i++){
        printf("%d ", output[i]);
    }
    printf("\n");

    printf("\nActualy Output: \n");
    for (int i = 0; i<KERNEL1X1_DST_SIZE; i++){
        printf("%d ", output_ref[i]);
    }
    printf("\n");
}

void kernel1x1_stride_x_y_arm_convolve_1x1_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[KERNEL1X1_STRIDE_X_Y_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = kernel1x1_stride_x_y_biases;
    const int8_t *input_data = kernel1x1_stride_x_y_input;
    const int8_t *output_ref = kernel1x1_stride_x_y_output_ref;
    const int32_t *output_ref_size = KERNEL1X1_STRIDE_X_Y_DST_SIZE;

    input_dims.n = KERNEL1X1_STRIDE_X_Y_INPUT_BATCHES;
    input_dims.h = KERNEL1X1_STRIDE_X_Y_INPUT_H;
    input_dims.w = KERNEL1X1_STRIDE_X_Y_INPUT_W;
    input_dims.c = KERNEL1X1_STRIDE_X_Y_IN_CH;

    filter_dims.n = KERNEL1X1_STRIDE_X_Y_OUT_CH;
    filter_dims.h = KERNEL1X1_STRIDE_X_Y_FILTER_Y;
    filter_dims.w = KERNEL1X1_STRIDE_X_Y_FILTER_X;
    filter_dims.c = KERNEL1X1_STRIDE_X_Y_IN_CH;

    output_dims.n = KERNEL1X1_STRIDE_X_Y_INPUT_BATCHES;
    output_dims.h = KERNEL1X1_STRIDE_X_Y_OUTPUT_H;
    output_dims.w = KERNEL1X1_STRIDE_X_Y_OUTPUT_W;
    output_dims.c = KERNEL1X1_STRIDE_X_Y_OUT_CH;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    conv_params.padding.w = KERNEL1X1_STRIDE_X_Y_PAD_X;
    conv_params.padding.h = KERNEL1X1_STRIDE_X_Y_PAD_Y;
    conv_params.stride.w = KERNEL1X1_STRIDE_X_Y_STRIDE_X;
    conv_params.stride.h = KERNEL1X1_STRIDE_X_Y_STRIDE_Y;

    conv_params.input_offset = KERNEL1X1_STRIDE_X_Y_INPUT_OFFSET;
    conv_params.output_offset = KERNEL1X1_STRIDE_X_Y_OUTPUT_OFFSET;
    conv_params.activation.min = KERNEL1X1_STRIDE_X_Y_OUT_ACTIVATION_MIN;
    conv_params.activation.max = KERNEL1X1_STRIDE_X_Y_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)kernel1x1_stride_x_y_output_mult;
    quant_params.shift = (int32_t *)kernel1x1_stride_x_y_output_shift;

    const int32_t buf_size =
        arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    arm_cmsis_nn_status result = arm_convolve_wrapper_s8(&ctx,
                                                         &conv_params,
                                                         &quant_params,
                                                         &input_dims,
                                                         input_data,
                                                         &filter_dims,
                                                         kernel1x1_stride_x_y_weights,
                                                         &bias_dims,
                                                         bias_data,
                                                         &output_dims,
                                                         output);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
        ctx.size = 0;
    }
    int flag;
    for (int i = 0; i<KERNEL1X1_STRIDE_X_Y_DST_SIZE; i++){
        if (output[i] != output_ref[i]){
            flag = 0;
        }
        else{
            flag = 1;
        }
    }

    if (flag == 1){
        printf("\nConvolve Passed\n");
    }

    printf("\nComputed Output: \n");
    for (int i = 0; i<KERNEL1X1_STRIDE_X_Y_DST_SIZE; i++){
        printf("%d ", output[i]);
    }
    printf("\n");

    printf("\nActualy Output: \n");
    for (int i = 0; i<KERNEL1X1_STRIDE_X_Y_DST_SIZE; i++){
        printf("%d ", output_ref[i]);
    }
    printf("\n");
}

void softmax_arm_softmax_s8(void)
{
    const int32_t num_rows = SOFTMAX_NUM_ROWS;
    const int32_t row_size = SOFTMAX_ROW_SIZE;
    const int32_t mult = SOFTMAX_INPUT_MULT;
    const int32_t shift = SOFTMAX_INPUT_LEFT_SHIFT;
    const int32_t diff_min = SOFTMAX_DIFF_MIN;
    const int8_t *input_data = softmax_input;
    int8_t output[SOFTMAX_DST_SIZE];
    
    arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output);

    int flag;
    
    for(int i = 0; i < SOFTMAX_DST_SIZE; i++){
        if(output[i] != softmax_output_ref[i]){
            flag = 0;
        }
        else{
            flag = 1;
        }
    }

    if(flag = 1){
        printf("\nSoftmax Test Passed\n");
    }

    printf("Computed Output: \n");
    for (int i = 0; i<SOFTMAX_DST_SIZE; i++){
        printf("%d ", output[i]);
    }
    printf("\n");

    printf("Expected Output \n");
    for (int i = 0; i<SOFTMAX_DST_SIZE; i++){
        printf("%d ", softmax_output_ref[i]);
    }
    printf("\n");
}

void softmax_20_neuron(void)
{
    const int32_t num_rows = SOFTMAX_20_NUM_ROWS;
    const int32_t row_size = SOFTMAX_20_ROW_SIZE;
    const int32_t mult = SOFTMAX_20_INPUT_MULT;
    const int32_t shift = SOFTMAX_20_INPUT_LEFT_SHIFT;
    const int32_t diff_min = SOFTMAX_20_DIFF_MIN;
    const int8_t *input_data = softmax_20_input;
    int8_t output[SOFTMAX_20_DST_SIZE];
    
    arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output);

    int flag = 0;
    
    for(int i = 0; i < SOFTMAX_20_DST_SIZE; i++){
        if(output[i] != softmax_20_output_ref[i]){
            flag = 0;
        }
        else{
            flag = 1;
        }
    }

    if(flag = 1){
        printf("\nSoftmax Test Passed\n");
    }

    printf("Computed Output: \n");
    for (int i = 0; i<SOFTMAX_20_DST_SIZE; i++){
        printf("%d, ", output[i]);
    }
    printf("\n");

    printf("Expected Output \n");
    for (int i = 0; i<SOFTMAX_20_DST_SIZE; i++){
        printf("%d ", softmax_20_output_ref[i]);
    }
    printf("\n");
}

int main(void){
    // maxpooling_arm_max_pool_s8();
    // maxpooling_2_arm_max_pool_s8();
    // maxpooling_4_arm_max_pool_s8();
    // basic_2_arm_convolve_s4();
    // softmax_arm_softmax_s8();
    // stride2pad1_arm_convolve_s8();
    // conv_2_arm_convolve_s8();
    // conv_3_arm_convolve_s8();
    // kernel1x1_arm_convolve_1x1_s8_fast();
    // kernel1x1_stride_x_y_arm_convolve_1x1_s8();
    maxpooling_20_neuron();
    softmax_20_neuron();
}
