#ifndef __DECLARACTION1_H__
#define __DECLARACTION1_H__

#include "../include/declaration1.cuh"

__global__ void forwardPass_conv2d_l50(float input[64][256][28][28], float output[64][256][28][28], float weight[256][256][3][3]);

__global__ void forwardPass_BatchNorm2d_l51(float input[64][256][28][28], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][28][28], float v2[256], float output[64][256][28][28]);

__global__ void forwardPass_conv2d_l52(float input[64][256][28][28], float output[64][512][28][28], float weight[512][256][1][1]);

__global__ void forwardPass_BatchNorm2d_l53(float input[64][512][28][28], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][28][28], float v2[512], float output[64][512][28][28]);

__global__ void forwardPass_ReLU_l54(float input[64][512][28][28], float output[64][512][28][28]);

__global__ void forwardPass_Add_l55(float* input0, float* input1, float* out, long N);

__global__ void forwardPass_conv2d_l56(float input[64][512][28][28], float output[64][256][28][28], float weight[256][512][1][1]);

__global__ void forwardPass_BatchNorm2d_l57(float input[64][256][28][28], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][28][28], float v2[256], float output[64][256][28][28]);

__global__ void forwardPass_conv2d_l58(float input[64][256][28][28], float output[64][256][28][28], float weight[256][256][3][3]);

__global__ void forwardPass_BatchNorm2d_l59(float input[64][256][28][28], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][28][28], float v2[256], float output[64][256][28][28]);

__global__ void forwardPass_conv2d_l60(float input[64][256][28][28], float output[64][512][28][28], float weight[512][256][1][1]);

__global__ void forwardPass_BatchNorm2d_l61(float input[64][512][28][28], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][28][28], float v2[512], float output[64][512][28][28]);

__global__ void forwardPass_ReLU_l62(float input[64][512][28][28], float output[64][512][28][28]);

__global__ void forwardPass_Add_l63(float* input0, float* input1, float* out, long N);

__global__ void forwardPass_conv2d_l64(float input[64][512][28][28], float output[64][512][28][28], float weight[512][512][1][1]);

__global__ void forwardPass_BatchNorm2d_l65(float input[64][512][28][28], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][28][28], float v2[512], float output[64][512][28][28]);

__global__ void forwardPass_conv2d_l66(float input[64][512][28][28], float output[64][512][14][14], float weight[512][512][3][3]);

__global__ void forwardPass_BatchNorm2d_l67(float input[64][512][14][14], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][14][14], float v2[512], float output[64][512][14][14]);

__global__ void forwardPass_conv2d_l68(float input[64][512][14][14], float output[64][1024][14][14], float weight[1024][512][1][1]);

__global__ void forwardPass_BatchNorm2d_l69(float input[64][1024][14][14], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][14][14], float v2[1024], float output[64][1024][14][14]);

__global__ void forwardPass_ReLU_l70(float input[64][1024][14][14], float output[64][1024][14][14]);

__global__ void forwardPass_conv2d_l71(float input[64][512][28][28], float output[64][1024][14][14], float weight[1024][512][1][1]);

__global__ void forwardPass_BatchNorm2d_l72(float input[64][1024][14][14], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][14][14], float v2[1024], float output[64][1024][14][14]);

__global__ void forwardPass_Add_l73(float* input0, float* input1, float* out, long N);

__global__ void forwardPass_conv2d_l74(float input[64][1024][14][14], float output[64][512][14][14], float weight[512][1024][1][1]);

__global__ void forwardPass_BatchNorm2d_l75(float input[64][512][14][14], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][14][14], float v2[512], float output[64][512][14][14]);

__global__ void forwardPass_conv2d_l76(float input[64][512][14][14], float output[64][512][14][14], float weight[512][512][3][3]);

__global__ void forwardPass_BatchNorm2d_l77(float input[64][512][14][14], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][14][14], float v2[512], float output[64][512][14][14]);

__global__ void forwardPass_conv2d_l78(float input[64][512][14][14], float output[64][1024][14][14], float weight[1024][512][1][1]);

__global__ void forwardPass_BatchNorm2d_l79(float input[64][1024][14][14], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][14][14], float v2[1024], float output[64][1024][14][14]);

__global__ void forwardPass_ReLU_l80(float input[64][1024][14][14], float output[64][1024][14][14]);

__global__ void forwardPass_Add_l81(float* input0, float* input1, float* out, long N);

__global__ void forwardPass_conv2d_l82(float input[64][1024][14][14], float output[64][512][14][14], float weight[512][1024][1][1]);

__global__ void forwardPass_BatchNorm2d_l83(float input[64][512][14][14], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][14][14], float v2[512], float output[64][512][14][14]);

__global__ void forwardPass_conv2d_l84(float input[64][512][14][14], float output[64][512][14][14], float weight[512][512][3][3]);

__global__ void forwardPass_BatchNorm2d_l85(float input[64][512][14][14], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][14][14], float v2[512], float output[64][512][14][14]);

__global__ void forwardPass_conv2d_l86(float input[64][512][14][14], float output[64][1024][14][14], float weight[1024][512][1][1]);

__global__ void forwardPass_BatchNorm2d_l87(float input[64][1024][14][14], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][14][14], float v2[1024], float output[64][1024][14][14]);

__global__ void forwardPass_ReLU_l88(float input[64][1024][14][14], float output[64][1024][14][14]);

__global__ void forwardPass_Add_l89(float* input0, float* input1, float* out, long N);

__global__ void forwardPass_conv2d_l90(float input[64][1024][14][14], float output[64][512][14][14], float weight[512][1024][1][1]);

__global__ void forwardPass_BatchNorm2d_l91(float input[64][512][14][14], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][14][14], float v2[512], float output[64][512][14][14]);

__global__ void forwardPass_conv2d_l92(float input[64][512][14][14], float output[64][512][14][14], float weight[512][512][3][3]);

__global__ void forwardPass_BatchNorm2d_l93(float input[64][512][14][14], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][14][14], float v2[512], float output[64][512][14][14]);

__global__ void forwardPass_conv2d_l94(float input[64][512][14][14], float output[64][1024][14][14], float weight[1024][512][1][1]);

__global__ void forwardPass_BatchNorm2d_l95(float input[64][1024][14][14], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][14][14], float v2[1024], float output[64][1024][14][14]);

__global__ void forwardPass_ReLU_l96(float input[64][1024][14][14], float output[64][1024][14][14]);

__global__ void forwardPass_Add_l97(float* input0, float* input1, float* out, long N);

__global__ void forwardPass_conv2d_l98(float input[64][1024][14][14], float output[64][512][14][14], float weight[512][1024][1][1]);

__global__ void forwardPass_BatchNorm2d_l99(float input[64][512][14][14], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][14][14], float v2[512], float output[64][512][14][14]);

#endif
