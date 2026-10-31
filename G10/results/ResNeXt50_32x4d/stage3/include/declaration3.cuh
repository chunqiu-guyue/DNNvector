#ifndef __DECLARACTION3_H__
#define __DECLARACTION3_H__

#include "../include/declaration3.cuh"

__global__ void backwardPass_BatchNorm2d_l135(float d_output[64][1024][7][7], float v1[64][1024][7][7], float v2[1024], float gamma_and_beta[2048], float dv1[64][1024][7][7], float dv2[1024], float dvar[1024], float dmu[1024], float d_input[64][1024][7][7], float d_gamma_and_beta[2048]);

__global__ void backwardPass_conv2d_Weight_l134(float input[64][1024][7][7], float d_output[64][1024][7][7], float d_weight[1024][1024][3][3]);

__global__ void backwardPass_conv2d_Input_l134(float d_input[64][1024][7][7], float d_output[64][1024][7][7], float weight[1024][1024][3][3]);

__global__ void backwardPass_BatchNorm2d_l133(float d_output[64][1024][7][7], float v1[64][1024][7][7], float v2[1024], float gamma_and_beta[2048], float dv1[64][1024][7][7], float dv2[1024], float dvar[1024], float dmu[1024], float d_input[64][1024][7][7], float d_gamma_and_beta[2048]);

__global__ void backwardPass_conv2d_Weight_l132(float input[64][2048][7][7], float d_output[64][1024][7][7], float d_weight[1024][2048][1][1]);

__global__ void backwardPass_conv2d_Input_l132(float d_input[64][2048][7][7], float d_output[64][1024][7][7], float weight[1024][2048][1][1]);

__global__ void multiGradient_Add_l131(float* d_output0, float* d_output1, long N);

__global__ void backwardPass_ReLU_l130(float d_input[64][2048][7][7], float input[64][2048][7][7], float d_output[64][2048][7][7]);

__global__ void backwardPass_BatchNorm2d_l129(float d_output[64][2048][7][7], float v1[64][2048][7][7], float v2[2048], float gamma_and_beta[4096], float dv1[64][2048][7][7], float dv2[2048], float dvar[2048], float dmu[2048], float d_input[64][2048][7][7], float d_gamma_and_beta[4096]);

__global__ void backwardPass_conv2d_Weight_l128(float input[64][1024][7][7], float d_output[64][2048][7][7], float d_weight[2048][1024][1][1]);

__global__ void backwardPass_conv2d_Input_l128(float d_input[64][1024][7][7], float d_output[64][2048][7][7], float weight[2048][1024][1][1]);

__global__ void backwardPass_BatchNorm2d_l127(float d_output[64][1024][7][7], float v1[64][1024][7][7], float v2[1024], float gamma_and_beta[2048], float dv1[64][1024][7][7], float dv2[1024], float dvar[1024], float dmu[1024], float d_input[64][1024][7][7], float d_gamma_and_beta[2048]);

__global__ void backwardPass_conv2d_Weight_l126(float input[64][1024][7][7], float d_output[64][1024][7][7], float d_weight[1024][1024][3][3]);

__global__ void backwardPass_conv2d_Input_l126(float d_input[64][1024][7][7], float d_output[64][1024][7][7], float weight[1024][1024][3][3]);

__global__ void backwardPass_BatchNorm2d_l125(float d_output[64][1024][7][7], float v1[64][1024][7][7], float v2[1024], float gamma_and_beta[2048], float dv1[64][1024][7][7], float dv2[1024], float dvar[1024], float dmu[1024], float d_input[64][1024][7][7], float d_gamma_and_beta[2048]);

__global__ void backwardPass_conv2d_Weight_l124(float input[64][2048][7][7], float d_output[64][1024][7][7], float d_weight[1024][2048][1][1]);

__global__ void backwardPass_conv2d_Input_l124(float d_input[64][2048][7][7], float d_output[64][1024][7][7], float weight[1024][2048][1][1]);

__global__ void multiGradient_Add_l123(float* d_output0, float* d_output1, long N);

__global__ void backwardPass_BatchNorm2d_l122(float d_output[64][2048][7][7], float v1[64][2048][7][7], float v2[2048], float gamma_and_beta[4096], float dv1[64][2048][7][7], float dv2[2048], float dvar[2048], float dmu[2048], float d_input[64][2048][7][7], float d_gamma_and_beta[4096]);

__global__ void backwardPass_conv2d_Weight_l121(float input[64][1024][14][14], float d_output[64][2048][7][7], float d_weight[2048][1024][1][1]);

__global__ void backwardPass_conv2d_Input_l121(float d_input[64][1024][14][14], float d_output[64][2048][7][7], float weight[2048][1024][1][1]);

__global__ void backwardPass_ReLU_l120(float d_input[64][2048][7][7], float input[64][2048][7][7], float d_output[64][2048][7][7]);

__global__ void backwardPass_BatchNorm2d_l119(float d_output[64][2048][7][7], float v1[64][2048][7][7], float v2[2048], float gamma_and_beta[4096], float dv1[64][2048][7][7], float dv2[2048], float dvar[2048], float dmu[2048], float d_input[64][2048][7][7], float d_gamma_and_beta[4096]);

__global__ void backwardPass_conv2d_Weight_l118(float input[64][1024][7][7], float d_output[64][2048][7][7], float d_weight[2048][1024][1][1]);

__global__ void backwardPass_conv2d_Input_l118(float d_input[64][1024][7][7], float d_output[64][2048][7][7], float weight[2048][1024][1][1]);

__global__ void backwardPass_BatchNorm2d_l117(float d_output[64][1024][7][7], float v1[64][1024][7][7], float v2[1024], float gamma_and_beta[2048], float dv1[64][1024][7][7], float dv2[1024], float dvar[1024], float dmu[1024], float d_input[64][1024][7][7], float d_gamma_and_beta[2048]);

__global__ void backwardPass_conv2d_Weight_l116(float input[64][1024][14][14], float d_output[64][1024][7][7], float d_weight[1024][1024][3][3]);

__global__ void backwardPass_conv2d_Input_l116(float d_input[64][1024][14][14], float d_output[64][1024][7][7], float weight[1024][1024][3][3]);

__global__ void backwardPass_BatchNorm2d_l115(float d_output[64][1024][14][14], float v1[64][1024][14][14], float v2[1024], float gamma_and_beta[2048], float dv1[64][1024][14][14], float dv2[1024], float dvar[1024], float dmu[1024], float d_input[64][1024][14][14], float d_gamma_and_beta[2048]);

__global__ void backwardPass_conv2d_Weight_l114(float input[64][1024][14][14], float d_output[64][1024][14][14], float d_weight[1024][1024][1][1]);

__global__ void backwardPass_conv2d_Input_l114(float d_input[64][1024][14][14], float d_output[64][1024][14][14], float weight[1024][1024][1][1]);

__global__ void multiGradient_Add_l113(float* d_output0, float* d_output1, long N);

__global__ void backwardPass_ReLU_l112(float d_input[64][1024][14][14], float input[64][1024][14][14], float d_output[64][1024][14][14]);

__global__ void backwardPass_BatchNorm2d_l111(float d_output[64][1024][14][14], float v1[64][1024][14][14], float v2[1024], float gamma_and_beta[2048], float dv1[64][1024][14][14], float dv2[1024], float dvar[1024], float dmu[1024], float d_input[64][1024][14][14], float d_gamma_and_beta[2048]);

__global__ void backwardPass_conv2d_Weight_l110(float input[64][512][14][14], float d_output[64][1024][14][14], float d_weight[1024][512][1][1]);

__global__ void backwardPass_conv2d_Input_l110(float d_input[64][512][14][14], float d_output[64][1024][14][14], float weight[1024][512][1][1]);

__global__ void backwardPass_BatchNorm2d_l109(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]);

__global__ void backwardPass_conv2d_Weight_l108(float input[64][512][14][14], float d_output[64][512][14][14], float d_weight[512][512][3][3]);

__global__ void backwardPass_conv2d_Input_l108(float d_input[64][512][14][14], float d_output[64][512][14][14], float weight[512][512][3][3]);

__global__ void backwardPass_BatchNorm2d_l107(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]);

__global__ void backwardPass_conv2d_Weight_l106(float input[64][1024][14][14], float d_output[64][512][14][14], float d_weight[512][1024][1][1]);

__global__ void backwardPass_conv2d_Input_l106(float d_input[64][1024][14][14], float d_output[64][512][14][14], float weight[512][1024][1][1]);

__global__ void multiGradient_Add_l105(float* d_output0, float* d_output1, long N);

__global__ void backwardPass_ReLU_l104(float d_input[64][1024][14][14], float input[64][1024][14][14], float d_output[64][1024][14][14]);

__global__ void backwardPass_BatchNorm2d_l103(float d_output[64][1024][14][14], float v1[64][1024][14][14], float v2[1024], float gamma_and_beta[2048], float dv1[64][1024][14][14], float dv2[1024], float dvar[1024], float dmu[1024], float d_input[64][1024][14][14], float d_gamma_and_beta[2048]);

__global__ void backwardPass_conv2d_Weight_l102(float input[64][512][14][14], float d_output[64][1024][14][14], float d_weight[1024][512][1][1]);

__global__ void backwardPass_conv2d_Input_l102(float d_input[64][512][14][14], float d_output[64][1024][14][14], float weight[1024][512][1][1]);

__global__ void backwardPass_BatchNorm2d_l101(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]);

__global__ void backwardPass_conv2d_Weight_l100(float input[64][512][14][14], float d_output[64][512][14][14], float d_weight[512][512][3][3]);

__global__ void backwardPass_conv2d_Input_l100(float d_input[64][512][14][14], float d_output[64][512][14][14], float weight[512][512][3][3]);

#endif
