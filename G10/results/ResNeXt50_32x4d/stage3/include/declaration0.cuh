#ifndef __DECLARACTION0_H__
#define __DECLARACTION0_H__

#include "../include/declaration0.cuh"

__global__ void forwardPass_conv2d_l0(float input[64][3][224][224], float output[64][64][112][112], float weight[64][3][7][7]);

__global__ void forwardPass_BatchNorm2d_l1(float input[64][64][112][112], float gamma_and_beta[128], float running_m[64], float running_v[64], float mu[64], float var[64], float v1[64][64][112][112], float v2[64], float output[64][64][112][112]);

__global__ void forwardPass_ReLU_l2(float input[64][64][112][112], float output[64][64][112][112]);

__global__ void forwardPass_MaxPool2d_l3(float input[64][64][112][112], float output[64][64][56][56]);

__global__ void forwardPass_conv2d_l4(float input[64][64][56][56], float output[64][128][56][56], float weight[128][64][1][1]);

__global__ void forwardPass_BatchNorm2d_l5(float input[64][128][56][56], float gamma_and_beta[256], float running_m[128], float running_v[128], float mu[128], float var[128], float v1[64][128][56][56], float v2[128], float output[64][128][56][56]);

__global__ void forwardPass_conv2d_l6(float input[64][128][56][56], float output[64][128][56][56], float weight[128][128][3][3]);

__global__ void forwardPass_BatchNorm2d_l7(float input[64][128][56][56], float gamma_and_beta[256], float running_m[128], float running_v[128], float mu[128], float var[128], float v1[64][128][56][56], float v2[128], float output[64][128][56][56]);

__global__ void forwardPass_conv2d_l8(float input[64][128][56][56], float output[64][256][56][56], float weight[256][128][1][1]);

__global__ void forwardPass_BatchNorm2d_l9(float input[64][256][56][56], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][56][56], float v2[256], float output[64][256][56][56]);

__global__ void forwardPass_ReLU_l10(float input[64][256][56][56], float output[64][256][56][56]);

__global__ void forwardPass_conv2d_l11(float input[64][64][56][56], float output[64][256][56][56], float weight[256][64][1][1]);

__global__ void forwardPass_BatchNorm2d_l12(float input[64][256][56][56], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][56][56], float v2[256], float output[64][256][56][56]);

__global__ void forwardPass_Add_l13(float* input0, float* input1, float* out, long N);

__global__ void forwardPass_conv2d_l14(float input[64][256][56][56], float output[64][128][56][56], float weight[128][256][1][1]);

__global__ void forwardPass_BatchNorm2d_l15(float input[64][128][56][56], float gamma_and_beta[256], float running_m[128], float running_v[128], float mu[128], float var[128], float v1[64][128][56][56], float v2[128], float output[64][128][56][56]);

__global__ void forwardPass_conv2d_l16(float input[64][128][56][56], float output[64][128][56][56], float weight[128][128][3][3]);

__global__ void forwardPass_BatchNorm2d_l17(float input[64][128][56][56], float gamma_and_beta[256], float running_m[128], float running_v[128], float mu[128], float var[128], float v1[64][128][56][56], float v2[128], float output[64][128][56][56]);

__global__ void forwardPass_conv2d_l18(float input[64][128][56][56], float output[64][256][56][56], float weight[256][128][1][1]);

__global__ void forwardPass_BatchNorm2d_l19(float input[64][256][56][56], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][56][56], float v2[256], float output[64][256][56][56]);

__global__ void forwardPass_ReLU_l20(float input[64][256][56][56], float output[64][256][56][56]);

__global__ void forwardPass_Add_l21(float* input0, float* input1, float* out, long N);

__global__ void forwardPass_conv2d_l22(float input[64][256][56][56], float output[64][128][56][56], float weight[128][256][1][1]);

__global__ void forwardPass_BatchNorm2d_l23(float input[64][128][56][56], float gamma_and_beta[256], float running_m[128], float running_v[128], float mu[128], float var[128], float v1[64][128][56][56], float v2[128], float output[64][128][56][56]);

__global__ void forwardPass_conv2d_l24(float input[64][128][56][56], float output[64][128][56][56], float weight[128][128][3][3]);

__global__ void forwardPass_BatchNorm2d_l25(float input[64][128][56][56], float gamma_and_beta[256], float running_m[128], float running_v[128], float mu[128], float var[128], float v1[64][128][56][56], float v2[128], float output[64][128][56][56]);

__global__ void forwardPass_conv2d_l26(float input[64][128][56][56], float output[64][256][56][56], float weight[256][128][1][1]);

__global__ void forwardPass_BatchNorm2d_l27(float input[64][256][56][56], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][56][56], float v2[256], float output[64][256][56][56]);

__global__ void forwardPass_ReLU_l28(float input[64][256][56][56], float output[64][256][56][56]);

__global__ void forwardPass_Add_l29(float* input0, float* input1, float* out, long N);

__global__ void forwardPass_conv2d_l30(float input[64][256][56][56], float output[64][256][56][56], float weight[256][256][1][1]);

__global__ void forwardPass_BatchNorm2d_l31(float input[64][256][56][56], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][56][56], float v2[256], float output[64][256][56][56]);

__global__ void forwardPass_conv2d_l32(float input[64][256][56][56], float output[64][256][28][28], float weight[256][256][3][3]);

__global__ void forwardPass_BatchNorm2d_l33(float input[64][256][28][28], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][28][28], float v2[256], float output[64][256][28][28]);

__global__ void forwardPass_conv2d_l34(float input[64][256][28][28], float output[64][512][28][28], float weight[512][256][1][1]);

__global__ void forwardPass_BatchNorm2d_l35(float input[64][512][28][28], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][28][28], float v2[512], float output[64][512][28][28]);

__global__ void forwardPass_ReLU_l36(float input[64][512][28][28], float output[64][512][28][28]);

__global__ void forwardPass_conv2d_l37(float input[64][256][56][56], float output[64][512][28][28], float weight[512][256][1][1]);

__global__ void forwardPass_BatchNorm2d_l38(float input[64][512][28][28], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][28][28], float v2[512], float output[64][512][28][28]);

__global__ void forwardPass_Add_l39(float* input0, float* input1, float* out, long N);

__global__ void forwardPass_conv2d_l40(float input[64][512][28][28], float output[64][256][28][28], float weight[256][512][1][1]);

__global__ void forwardPass_BatchNorm2d_l41(float input[64][256][28][28], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][28][28], float v2[256], float output[64][256][28][28]);

__global__ void forwardPass_conv2d_l42(float input[64][256][28][28], float output[64][256][28][28], float weight[256][256][3][3]);

__global__ void forwardPass_BatchNorm2d_l43(float input[64][256][28][28], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][28][28], float v2[256], float output[64][256][28][28]);

__global__ void forwardPass_conv2d_l44(float input[64][256][28][28], float output[64][512][28][28], float weight[512][256][1][1]);

__global__ void forwardPass_BatchNorm2d_l45(float input[64][512][28][28], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][28][28], float v2[512], float output[64][512][28][28]);

__global__ void forwardPass_ReLU_l46(float input[64][512][28][28], float output[64][512][28][28]);

__global__ void forwardPass_Add_l47(float* input0, float* input1, float* out, long N);

__global__ void forwardPass_conv2d_l48(float input[64][512][28][28], float output[64][256][28][28], float weight[256][512][1][1]);

__global__ void forwardPass_BatchNorm2d_l49(float input[64][256][28][28], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][28][28], float v2[256], float output[64][256][28][28]);

#endif
