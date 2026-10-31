#ifndef __DECLARACTION6_H__
#define __DECLARACTION6_H__

#include "../include/declaration6.cuh"

__global__ void backwardPass_conv2d_Weight_l26(float input[64][128][56][56], float d_output[64][256][56][56], float d_weight[256][128][1][1]);

__global__ void backwardPass_conv2d_Input_l26(float d_input[64][128][56][56], float d_output[64][256][56][56], float weight[256][128][1][1]);

__global__ void backwardPass_BatchNorm2d_l25(float d_output[64][128][56][56], float v1[64][128][56][56], float v2[128], float gamma_and_beta[256], float dv1[64][128][56][56], float dv2[128], float dvar[128], float dmu[128], float d_input[64][128][56][56], float d_gamma_and_beta[256]);

__global__ void backwardPass_conv2d_Weight_l24(float input[64][128][56][56], float d_output[64][128][56][56], float d_weight[128][128][3][3]);

__global__ void backwardPass_conv2d_Input_l24(float d_input[64][128][56][56], float d_output[64][128][56][56], float weight[128][128][3][3]);

__global__ void backwardPass_BatchNorm2d_l23(float d_output[64][128][56][56], float v1[64][128][56][56], float v2[128], float gamma_and_beta[256], float dv1[64][128][56][56], float dv2[128], float dvar[128], float dmu[128], float d_input[64][128][56][56], float d_gamma_and_beta[256]);

__global__ void backwardPass_conv2d_Weight_l22(float input[64][256][56][56], float d_output[64][128][56][56], float d_weight[128][256][1][1]);

__global__ void backwardPass_conv2d_Input_l22(float d_input[64][256][56][56], float d_output[64][128][56][56], float weight[128][256][1][1]);

__global__ void multiGradient_Add_l21(float* d_output0, float* d_output1, long N);

__global__ void backwardPass_ReLU_l20(float d_input[64][256][56][56], float input[64][256][56][56], float d_output[64][256][56][56]);

__global__ void backwardPass_BatchNorm2d_l19(float d_output[64][256][56][56], float v1[64][256][56][56], float v2[256], float gamma_and_beta[512], float dv1[64][256][56][56], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][56][56], float d_gamma_and_beta[512]);

__global__ void backwardPass_conv2d_Weight_l18(float input[64][128][56][56], float d_output[64][256][56][56], float d_weight[256][128][1][1]);

__global__ void backwardPass_conv2d_Input_l18(float d_input[64][128][56][56], float d_output[64][256][56][56], float weight[256][128][1][1]);

__global__ void backwardPass_BatchNorm2d_l17(float d_output[64][128][56][56], float v1[64][128][56][56], float v2[128], float gamma_and_beta[256], float dv1[64][128][56][56], float dv2[128], float dvar[128], float dmu[128], float d_input[64][128][56][56], float d_gamma_and_beta[256]);

__global__ void backwardPass_conv2d_Weight_l16(float input[64][128][56][56], float d_output[64][128][56][56], float d_weight[128][128][3][3]);

__global__ void backwardPass_conv2d_Input_l16(float d_input[64][128][56][56], float d_output[64][128][56][56], float weight[128][128][3][3]);

__global__ void backwardPass_BatchNorm2d_l15(float d_output[64][128][56][56], float v1[64][128][56][56], float v2[128], float gamma_and_beta[256], float dv1[64][128][56][56], float dv2[128], float dvar[128], float dmu[128], float d_input[64][128][56][56], float d_gamma_and_beta[256]);

__global__ void backwardPass_conv2d_Weight_l14(float input[64][256][56][56], float d_output[64][128][56][56], float d_weight[128][256][1][1]);

__global__ void backwardPass_conv2d_Input_l14(float d_input[64][256][56][56], float d_output[64][128][56][56], float weight[128][256][1][1]);

__global__ void multiGradient_Add_l13(float* d_output0, float* d_output1, long N);

__global__ void backwardPass_BatchNorm2d_l12(float d_output[64][256][56][56], float v1[64][256][56][56], float v2[256], float gamma_and_beta[512], float dv1[64][256][56][56], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][56][56], float d_gamma_and_beta[512]);

__global__ void backwardPass_conv2d_Weight_l11(float input[64][64][56][56], float d_output[64][256][56][56], float d_weight[256][64][1][1]);

__global__ void backwardPass_conv2d_Input_l11(float d_input[64][64][56][56], float d_output[64][256][56][56], float weight[256][64][1][1]);

__global__ void backwardPass_ReLU_l10(float d_input[64][256][56][56], float input[64][256][56][56], float d_output[64][256][56][56]);

__global__ void backwardPass_BatchNorm2d_l9(float d_output[64][256][56][56], float v1[64][256][56][56], float v2[256], float gamma_and_beta[512], float dv1[64][256][56][56], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][56][56], float d_gamma_and_beta[512]);

__global__ void backwardPass_conv2d_Weight_l8(float input[64][128][56][56], float d_output[64][256][56][56], float d_weight[256][128][1][1]);

__global__ void backwardPass_conv2d_Input_l8(float d_input[64][128][56][56], float d_output[64][256][56][56], float weight[256][128][1][1]);

__global__ void backwardPass_BatchNorm2d_l7(float d_output[64][128][56][56], float v1[64][128][56][56], float v2[128], float gamma_and_beta[256], float dv1[64][128][56][56], float dv2[128], float dvar[128], float dmu[128], float d_input[64][128][56][56], float d_gamma_and_beta[256]);

__global__ void backwardPass_conv2d_Weight_l6(float input[64][128][56][56], float d_output[64][128][56][56], float d_weight[128][128][3][3]);

__global__ void backwardPass_conv2d_Input_l6(float d_input[64][128][56][56], float d_output[64][128][56][56], float weight[128][128][3][3]);

__global__ void backwardPass_BatchNorm2d_l5(float d_output[64][128][56][56], float v1[64][128][56][56], float v2[128], float gamma_and_beta[256], float dv1[64][128][56][56], float dv2[128], float dvar[128], float dmu[128], float d_input[64][128][56][56], float d_gamma_and_beta[256]);

__global__ void backwardPass_conv2d_Weight_l4(float input[64][64][56][56], float d_output[64][128][56][56], float d_weight[128][64][1][1]);

__global__ void backwardPass_conv2d_Input_l4(float d_input[64][64][56][56], float d_output[64][128][56][56], float weight[128][64][1][1]);

__global__ void multiGradient_Add_l3(float* d_output0, float* d_output1, long N);

__global__ void backwardPass_MaxPool2d_l3(float d_input[64][64][112][112], float input[64][64][112][112], float d_output[64][64][56][56]);

__global__ void backwardPass_ReLU_l2(float d_input[64][64][112][112], float input[64][64][112][112], float d_output[64][64][112][112]);

__global__ void backwardPass_BatchNorm2d_l1(float d_output[64][64][112][112], float v1[64][64][112][112], float v2[64], float gamma_and_beta[128], float dv1[64][64][112][112], float dv2[64], float dvar[64], float dmu[64], float d_input[64][64][112][112], float d_gamma_and_beta[128]);

__global__ void backwardPass_conv2d_Weight_l0(float input[64][3][224][224], float d_output[64][64][112][112], float d_weight[64][3][7][7]);

__global__ void backwardPass_conv2d_Input_l0(float d_input[64][3][224][224], float d_output[64][64][112][112], float weight[64][3][7][7]);

#endif
