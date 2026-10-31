#ifndef __DECLARACTION2_H__
#define __DECLARACTION2_H__

#include "../include/declaration2.cuh"

__global__ void forwardPass_conv2d_l100(float input[64][512][14][14], float output[64][512][14][14], float weight[512][512][3][3]);

__global__ void forwardPass_BatchNorm2d_l101(float input[64][512][14][14], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][14][14], float v2[512], float output[64][512][14][14]);

__global__ void forwardPass_conv2d_l102(float input[64][512][14][14], float output[64][1024][14][14], float weight[1024][512][1][1]);

__global__ void forwardPass_BatchNorm2d_l103(float input[64][1024][14][14], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][14][14], float v2[1024], float output[64][1024][14][14]);

__global__ void forwardPass_ReLU_l104(float input[64][1024][14][14], float output[64][1024][14][14]);

__global__ void forwardPass_Add_l105(float* input0, float* input1, float* out, long N);

__global__ void forwardPass_conv2d_l106(float input[64][1024][14][14], float output[64][512][14][14], float weight[512][1024][1][1]);

__global__ void forwardPass_BatchNorm2d_l107(float input[64][512][14][14], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][14][14], float v2[512], float output[64][512][14][14]);

__global__ void forwardPass_conv2d_l108(float input[64][512][14][14], float output[64][512][14][14], float weight[512][512][3][3]);

__global__ void forwardPass_BatchNorm2d_l109(float input[64][512][14][14], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][14][14], float v2[512], float output[64][512][14][14]);

__global__ void forwardPass_conv2d_l110(float input[64][512][14][14], float output[64][1024][14][14], float weight[1024][512][1][1]);

__global__ void forwardPass_BatchNorm2d_l111(float input[64][1024][14][14], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][14][14], float v2[1024], float output[64][1024][14][14]);

__global__ void forwardPass_ReLU_l112(float input[64][1024][14][14], float output[64][1024][14][14]);

__global__ void forwardPass_Add_l113(float* input0, float* input1, float* out, long N);

__global__ void forwardPass_conv2d_l114(float input[64][1024][14][14], float output[64][1024][14][14], float weight[1024][1024][1][1]);

__global__ void forwardPass_BatchNorm2d_l115(float input[64][1024][14][14], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][14][14], float v2[1024], float output[64][1024][14][14]);

__global__ void forwardPass_conv2d_l116(float input[64][1024][14][14], float output[64][1024][7][7], float weight[1024][1024][3][3]);

__global__ void forwardPass_BatchNorm2d_l117(float input[64][1024][7][7], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][7][7], float v2[1024], float output[64][1024][7][7]);

__global__ void forwardPass_conv2d_l118(float input[64][1024][7][7], float output[64][2048][7][7], float weight[2048][1024][1][1]);

__global__ void forwardPass_BatchNorm2d_l119(float input[64][2048][7][7], float gamma_and_beta[4096], float running_m[2048], float running_v[2048], float mu[2048], float var[2048], float v1[64][2048][7][7], float v2[2048], float output[64][2048][7][7]);

__global__ void forwardPass_ReLU_l120(float input[64][2048][7][7], float output[64][2048][7][7]);

__global__ void forwardPass_conv2d_l121(float input[64][1024][14][14], float output[64][2048][7][7], float weight[2048][1024][1][1]);

__global__ void forwardPass_BatchNorm2d_l122(float input[64][2048][7][7], float gamma_and_beta[4096], float running_m[2048], float running_v[2048], float mu[2048], float var[2048], float v1[64][2048][7][7], float v2[2048], float output[64][2048][7][7]);

__global__ void forwardPass_Add_l123(float* input0, float* input1, float* out, long N);

__global__ void forwardPass_conv2d_l124(float input[64][2048][7][7], float output[64][1024][7][7], float weight[1024][2048][1][1]);

__global__ void forwardPass_BatchNorm2d_l125(float input[64][1024][7][7], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][7][7], float v2[1024], float output[64][1024][7][7]);

__global__ void forwardPass_conv2d_l126(float input[64][1024][7][7], float output[64][1024][7][7], float weight[1024][1024][3][3]);

__global__ void forwardPass_BatchNorm2d_l127(float input[64][1024][7][7], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][7][7], float v2[1024], float output[64][1024][7][7]);

__global__ void forwardPass_conv2d_l128(float input[64][1024][7][7], float output[64][2048][7][7], float weight[2048][1024][1][1]);

__global__ void forwardPass_BatchNorm2d_l129(float input[64][2048][7][7], float gamma_and_beta[4096], float running_m[2048], float running_v[2048], float mu[2048], float var[2048], float v1[64][2048][7][7], float v2[2048], float output[64][2048][7][7]);

__global__ void forwardPass_ReLU_l130(float input[64][2048][7][7], float output[64][2048][7][7]);

__global__ void forwardPass_Add_l131(float* input0, float* input1, float* out, long N);

__global__ void forwardPass_conv2d_l132(float input[64][2048][7][7], float output[64][1024][7][7], float weight[1024][2048][1][1]);

__global__ void forwardPass_BatchNorm2d_l133(float input[64][1024][7][7], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][7][7], float v2[1024], float output[64][1024][7][7]);

__global__ void forwardPass_conv2d_l134(float input[64][1024][7][7], float output[64][1024][7][7], float weight[1024][1024][3][3]);

__global__ void forwardPass_BatchNorm2d_l135(float input[64][1024][7][7], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][7][7], float v2[1024], float output[64][1024][7][7]);

__global__ void forwardPass_conv2d_l136(float input[64][1024][7][7], float output[64][2048][7][7], float weight[2048][1024][1][1]);

__global__ void forwardPass_BatchNorm2d_l137(float input[64][2048][7][7], float gamma_and_beta[4096], float running_m[2048], float running_v[2048], float mu[2048], float var[2048], float v1[64][2048][7][7], float v2[2048], float output[64][2048][7][7]);

__global__ void forwardPass_ReLU_l138(float input[64][2048][7][7], float output[64][2048][7][7]);

__global__ void forwardPass_Add_l139(float* input0, float* input1, float* out, long N);

__global__ void forwardPass_avgAdaptivePool2d_l140(float input[64][2048][7][7], float output[64][2048][1][1]);

__global__ void forwardPass_Linear_l141(float input[64][2048], float output[64][1000], float weight[2048][1000], float bias[1000]);

__global__ void backwardPass_Linear_bias_l141(float d_output[64][1000], float d_bias[1000]);

__global__ void backwardPass_Linear_weight_l141(float input[64][2048], float d_output[64][1000], float d_weight[2048][1000]);

__global__ void backwardPass_Linear_input_l141(float d_input[64][2048], float d_output[64][1000], float weight[2048][1000]);

__global__ void backwardPass_avgAdaptivePool2d_l140(float d_input[64][2048][7][7], float d_output[64][2048][1][1]);

__global__ void backwardPass_ReLU_l138(float d_input[64][2048][7][7], float input[64][2048][7][7], float d_output[64][2048][7][7]);

__global__ void backwardPass_BatchNorm2d_l137(float d_output[64][2048][7][7], float v1[64][2048][7][7], float v2[2048], float gamma_and_beta[4096], float dv1[64][2048][7][7], float dv2[2048], float dvar[2048], float dmu[2048], float d_input[64][2048][7][7], float d_gamma_and_beta[4096]);

__global__ void backwardPass_conv2d_Weight_l136(float input[64][1024][7][7], float d_output[64][2048][7][7], float d_weight[2048][1024][1][1]);

__global__ void backwardPass_conv2d_Input_l136(float d_input[64][1024][7][7], float d_output[64][2048][7][7], float weight[2048][1024][1][1]);

#endif
