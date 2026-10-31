#ifndef __DECLARACTION5_H__
#define __DECLARACTION5_H__

#include "../include/declaration5.cuh"

__global__ void multiGradient_Add_l63(float* d_output0, float* d_output1, long N);

__global__ void backwardPass_ReLU_l62(float d_input[64][512][28][28], float input[64][512][28][28], float d_output[64][512][28][28]);

__global__ void backwardPass_BatchNorm2d_l61(float d_output[64][512][28][28], float v1[64][512][28][28], float v2[512], float gamma_and_beta[1024], float dv1[64][512][28][28], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][28][28], float d_gamma_and_beta[1024]);

__global__ void backwardPass_conv2d_Weight_l60(float input[64][256][28][28], float d_output[64][512][28][28], float d_weight[512][256][1][1]);

__global__ void backwardPass_conv2d_Input_l60(float d_input[64][256][28][28], float d_output[64][512][28][28], float weight[512][256][1][1]);

__global__ void backwardPass_BatchNorm2d_l59(float d_output[64][256][28][28], float v1[64][256][28][28], float v2[256], float gamma_and_beta[512], float dv1[64][256][28][28], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][28][28], float d_gamma_and_beta[512]);

__global__ void backwardPass_conv2d_Weight_l58(float input[64][256][28][28], float d_output[64][256][28][28], float d_weight[256][256][3][3]);

__global__ void backwardPass_conv2d_Input_l58(float d_input[64][256][28][28], float d_output[64][256][28][28], float weight[256][256][3][3]);

__global__ void backwardPass_BatchNorm2d_l57(float d_output[64][256][28][28], float v1[64][256][28][28], float v2[256], float gamma_and_beta[512], float dv1[64][256][28][28], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][28][28], float d_gamma_and_beta[512]);

__global__ void backwardPass_conv2d_Weight_l56(float input[64][512][28][28], float d_output[64][256][28][28], float d_weight[256][512][1][1]);

__global__ void backwardPass_conv2d_Input_l56(float d_input[64][512][28][28], float d_output[64][256][28][28], float weight[256][512][1][1]);

__global__ void multiGradient_Add_l55(float* d_output0, float* d_output1, long N);

__global__ void backwardPass_ReLU_l54(float d_input[64][512][28][28], float input[64][512][28][28], float d_output[64][512][28][28]);

__global__ void backwardPass_BatchNorm2d_l53(float d_output[64][512][28][28], float v1[64][512][28][28], float v2[512], float gamma_and_beta[1024], float dv1[64][512][28][28], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][28][28], float d_gamma_and_beta[1024]);

__global__ void backwardPass_conv2d_Weight_l52(float input[64][256][28][28], float d_output[64][512][28][28], float d_weight[512][256][1][1]);

__global__ void backwardPass_conv2d_Input_l52(float d_input[64][256][28][28], float d_output[64][512][28][28], float weight[512][256][1][1]);

__global__ void backwardPass_BatchNorm2d_l51(float d_output[64][256][28][28], float v1[64][256][28][28], float v2[256], float gamma_and_beta[512], float dv1[64][256][28][28], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][28][28], float d_gamma_and_beta[512]);

__global__ void backwardPass_conv2d_Weight_l50(float input[64][256][28][28], float d_output[64][256][28][28], float d_weight[256][256][3][3]);

__global__ void backwardPass_conv2d_Input_l50(float d_input[64][256][28][28], float d_output[64][256][28][28], float weight[256][256][3][3]);

__global__ void backwardPass_BatchNorm2d_l49(float d_output[64][256][28][28], float v1[64][256][28][28], float v2[256], float gamma_and_beta[512], float dv1[64][256][28][28], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][28][28], float d_gamma_and_beta[512]);

__global__ void backwardPass_conv2d_Weight_l48(float input[64][512][28][28], float d_output[64][256][28][28], float d_weight[256][512][1][1]);

__global__ void backwardPass_conv2d_Input_l48(float d_input[64][512][28][28], float d_output[64][256][28][28], float weight[256][512][1][1]);

__global__ void multiGradient_Add_l47(float* d_output0, float* d_output1, long N);

__global__ void backwardPass_ReLU_l46(float d_input[64][512][28][28], float input[64][512][28][28], float d_output[64][512][28][28]);

__global__ void backwardPass_BatchNorm2d_l45(float d_output[64][512][28][28], float v1[64][512][28][28], float v2[512], float gamma_and_beta[1024], float dv1[64][512][28][28], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][28][28], float d_gamma_and_beta[1024]);

__global__ void backwardPass_conv2d_Weight_l44(float input[64][256][28][28], float d_output[64][512][28][28], float d_weight[512][256][1][1]);

__global__ void backwardPass_conv2d_Input_l44(float d_input[64][256][28][28], float d_output[64][512][28][28], float weight[512][256][1][1]);

__global__ void backwardPass_BatchNorm2d_l43(float d_output[64][256][28][28], float v1[64][256][28][28], float v2[256], float gamma_and_beta[512], float dv1[64][256][28][28], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][28][28], float d_gamma_and_beta[512]);

__global__ void backwardPass_conv2d_Weight_l42(float input[64][256][28][28], float d_output[64][256][28][28], float d_weight[256][256][3][3]);

__global__ void backwardPass_conv2d_Input_l42(float d_input[64][256][28][28], float d_output[64][256][28][28], float weight[256][256][3][3]);

__global__ void backwardPass_BatchNorm2d_l41(float d_output[64][256][28][28], float v1[64][256][28][28], float v2[256], float gamma_and_beta[512], float dv1[64][256][28][28], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][28][28], float d_gamma_and_beta[512]);

__global__ void backwardPass_conv2d_Weight_l40(float input[64][512][28][28], float d_output[64][256][28][28], float d_weight[256][512][1][1]);

__global__ void backwardPass_conv2d_Input_l40(float d_input[64][512][28][28], float d_output[64][256][28][28], float weight[256][512][1][1]);

__global__ void multiGradient_Add_l39(float* d_output0, float* d_output1, long N);

__global__ void backwardPass_BatchNorm2d_l38(float d_output[64][512][28][28], float v1[64][512][28][28], float v2[512], float gamma_and_beta[1024], float dv1[64][512][28][28], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][28][28], float d_gamma_and_beta[1024]);

__global__ void backwardPass_conv2d_Weight_l37(float input[64][256][56][56], float d_output[64][512][28][28], float d_weight[512][256][1][1]);

__global__ void backwardPass_conv2d_Input_l37(float d_input[64][256][56][56], float d_output[64][512][28][28], float weight[512][256][1][1]);

__global__ void backwardPass_ReLU_l36(float d_input[64][512][28][28], float input[64][512][28][28], float d_output[64][512][28][28]);

__global__ void backwardPass_BatchNorm2d_l35(float d_output[64][512][28][28], float v1[64][512][28][28], float v2[512], float gamma_and_beta[1024], float dv1[64][512][28][28], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][28][28], float d_gamma_and_beta[1024]);

__global__ void backwardPass_conv2d_Weight_l34(float input[64][256][28][28], float d_output[64][512][28][28], float d_weight[512][256][1][1]);

__global__ void backwardPass_conv2d_Input_l34(float d_input[64][256][28][28], float d_output[64][512][28][28], float weight[512][256][1][1]);

__global__ void backwardPass_BatchNorm2d_l33(float d_output[64][256][28][28], float v1[64][256][28][28], float v2[256], float gamma_and_beta[512], float dv1[64][256][28][28], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][28][28], float d_gamma_and_beta[512]);

__global__ void backwardPass_conv2d_Weight_l32(float input[64][256][56][56], float d_output[64][256][28][28], float d_weight[256][256][3][3]);

__global__ void backwardPass_conv2d_Input_l32(float d_input[64][256][56][56], float d_output[64][256][28][28], float weight[256][256][3][3]);

__global__ void backwardPass_BatchNorm2d_l31(float d_output[64][256][56][56], float v1[64][256][56][56], float v2[256], float gamma_and_beta[512], float dv1[64][256][56][56], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][56][56], float d_gamma_and_beta[512]);

__global__ void backwardPass_conv2d_Weight_l30(float input[64][256][56][56], float d_output[64][256][56][56], float d_weight[256][256][1][1]);

__global__ void backwardPass_conv2d_Input_l30(float d_input[64][256][56][56], float d_output[64][256][56][56], float weight[256][256][1][1]);

__global__ void multiGradient_Add_l29(float* d_output0, float* d_output1, long N);

__global__ void backwardPass_ReLU_l28(float d_input[64][256][56][56], float input[64][256][56][56], float d_output[64][256][56][56]);

__global__ void backwardPass_BatchNorm2d_l27(float d_output[64][256][56][56], float v1[64][256][56][56], float v2[256], float gamma_and_beta[512], float dv1[64][256][56][56], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][56][56], float d_gamma_and_beta[512]);

#endif
