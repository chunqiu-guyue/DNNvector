#ifndef __DECLARACTION4_H__
#define __DECLARACTION4_H__

#include "../include/declaration4.cuh"

__global__ void backwardPass_BatchNorm2d_l99(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]);

__global__ void backwardPass_conv2d_Weight_l98(float input[64][1024][14][14], float d_output[64][512][14][14], float d_weight[512][1024][1][1]);

__global__ void backwardPass_conv2d_Input_l98(float d_input[64][1024][14][14], float d_output[64][512][14][14], float weight[512][1024][1][1]);

__global__ void multiGradient_Add_l97(float* d_output0, float* d_output1, long N);

__global__ void backwardPass_ReLU_l96(float d_input[64][1024][14][14], float input[64][1024][14][14], float d_output[64][1024][14][14]);

__global__ void backwardPass_BatchNorm2d_l95(float d_output[64][1024][14][14], float v1[64][1024][14][14], float v2[1024], float gamma_and_beta[2048], float dv1[64][1024][14][14], float dv2[1024], float dvar[1024], float dmu[1024], float d_input[64][1024][14][14], float d_gamma_and_beta[2048]);

__global__ void backwardPass_conv2d_Weight_l94(float input[64][512][14][14], float d_output[64][1024][14][14], float d_weight[1024][512][1][1]);

__global__ void backwardPass_conv2d_Input_l94(float d_input[64][512][14][14], float d_output[64][1024][14][14], float weight[1024][512][1][1]);

__global__ void backwardPass_BatchNorm2d_l93(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]);

__global__ void backwardPass_conv2d_Weight_l92(float input[64][512][14][14], float d_output[64][512][14][14], float d_weight[512][512][3][3]);

__global__ void backwardPass_conv2d_Input_l92(float d_input[64][512][14][14], float d_output[64][512][14][14], float weight[512][512][3][3]);

__global__ void backwardPass_BatchNorm2d_l91(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]);

__global__ void backwardPass_conv2d_Weight_l90(float input[64][1024][14][14], float d_output[64][512][14][14], float d_weight[512][1024][1][1]);

__global__ void backwardPass_conv2d_Input_l90(float d_input[64][1024][14][14], float d_output[64][512][14][14], float weight[512][1024][1][1]);

__global__ void multiGradient_Add_l89(float* d_output0, float* d_output1, long N);

__global__ void backwardPass_ReLU_l88(float d_input[64][1024][14][14], float input[64][1024][14][14], float d_output[64][1024][14][14]);

__global__ void backwardPass_BatchNorm2d_l87(float d_output[64][1024][14][14], float v1[64][1024][14][14], float v2[1024], float gamma_and_beta[2048], float dv1[64][1024][14][14], float dv2[1024], float dvar[1024], float dmu[1024], float d_input[64][1024][14][14], float d_gamma_and_beta[2048]);

__global__ void backwardPass_conv2d_Weight_l86(float input[64][512][14][14], float d_output[64][1024][14][14], float d_weight[1024][512][1][1]);

__global__ void backwardPass_conv2d_Input_l86(float d_input[64][512][14][14], float d_output[64][1024][14][14], float weight[1024][512][1][1]);

__global__ void backwardPass_BatchNorm2d_l85(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]);

__global__ void backwardPass_conv2d_Weight_l84(float input[64][512][14][14], float d_output[64][512][14][14], float d_weight[512][512][3][3]);

__global__ void backwardPass_conv2d_Input_l84(float d_input[64][512][14][14], float d_output[64][512][14][14], float weight[512][512][3][3]);

__global__ void backwardPass_BatchNorm2d_l83(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]);

__global__ void backwardPass_conv2d_Weight_l82(float input[64][1024][14][14], float d_output[64][512][14][14], float d_weight[512][1024][1][1]);

__global__ void backwardPass_conv2d_Input_l82(float d_input[64][1024][14][14], float d_output[64][512][14][14], float weight[512][1024][1][1]);

__global__ void multiGradient_Add_l81(float* d_output0, float* d_output1, long N);

__global__ void backwardPass_ReLU_l80(float d_input[64][1024][14][14], float input[64][1024][14][14], float d_output[64][1024][14][14]);

__global__ void backwardPass_BatchNorm2d_l79(float d_output[64][1024][14][14], float v1[64][1024][14][14], float v2[1024], float gamma_and_beta[2048], float dv1[64][1024][14][14], float dv2[1024], float dvar[1024], float dmu[1024], float d_input[64][1024][14][14], float d_gamma_and_beta[2048]);

__global__ void backwardPass_conv2d_Weight_l78(float input[64][512][14][14], float d_output[64][1024][14][14], float d_weight[1024][512][1][1]);

__global__ void backwardPass_conv2d_Input_l78(float d_input[64][512][14][14], float d_output[64][1024][14][14], float weight[1024][512][1][1]);

__global__ void backwardPass_BatchNorm2d_l77(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]);

__global__ void backwardPass_conv2d_Weight_l76(float input[64][512][14][14], float d_output[64][512][14][14], float d_weight[512][512][3][3]);

__global__ void backwardPass_conv2d_Input_l76(float d_input[64][512][14][14], float d_output[64][512][14][14], float weight[512][512][3][3]);

__global__ void backwardPass_BatchNorm2d_l75(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]);

__global__ void backwardPass_conv2d_Weight_l74(float input[64][1024][14][14], float d_output[64][512][14][14], float d_weight[512][1024][1][1]);

__global__ void backwardPass_conv2d_Input_l74(float d_input[64][1024][14][14], float d_output[64][512][14][14], float weight[512][1024][1][1]);

__global__ void multiGradient_Add_l73(float* d_output0, float* d_output1, long N);

__global__ void backwardPass_BatchNorm2d_l72(float d_output[64][1024][14][14], float v1[64][1024][14][14], float v2[1024], float gamma_and_beta[2048], float dv1[64][1024][14][14], float dv2[1024], float dvar[1024], float dmu[1024], float d_input[64][1024][14][14], float d_gamma_and_beta[2048]);

__global__ void backwardPass_conv2d_Weight_l71(float input[64][512][28][28], float d_output[64][1024][14][14], float d_weight[1024][512][1][1]);

__global__ void backwardPass_conv2d_Input_l71(float d_input[64][512][28][28], float d_output[64][1024][14][14], float weight[1024][512][1][1]);

__global__ void backwardPass_ReLU_l70(float d_input[64][1024][14][14], float input[64][1024][14][14], float d_output[64][1024][14][14]);

__global__ void backwardPass_BatchNorm2d_l69(float d_output[64][1024][14][14], float v1[64][1024][14][14], float v2[1024], float gamma_and_beta[2048], float dv1[64][1024][14][14], float dv2[1024], float dvar[1024], float dmu[1024], float d_input[64][1024][14][14], float d_gamma_and_beta[2048]);

__global__ void backwardPass_conv2d_Weight_l68(float input[64][512][14][14], float d_output[64][1024][14][14], float d_weight[1024][512][1][1]);

__global__ void backwardPass_conv2d_Input_l68(float d_input[64][512][14][14], float d_output[64][1024][14][14], float weight[1024][512][1][1]);

__global__ void backwardPass_BatchNorm2d_l67(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]);

__global__ void backwardPass_conv2d_Weight_l66(float input[64][512][28][28], float d_output[64][512][14][14], float d_weight[512][512][3][3]);

__global__ void backwardPass_conv2d_Input_l66(float d_input[64][512][28][28], float d_output[64][512][14][14], float weight[512][512][3][3]);

__global__ void backwardPass_BatchNorm2d_l65(float d_output[64][512][28][28], float v1[64][512][28][28], float v2[512], float gamma_and_beta[1024], float dv1[64][512][28][28], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][28][28], float d_gamma_and_beta[1024]);

__global__ void backwardPass_conv2d_Weight_l64(float input[64][512][28][28], float d_output[64][512][28][28], float d_weight[512][512][1][1]);

__global__ void backwardPass_conv2d_Input_l64(float d_input[64][512][28][28], float d_output[64][512][28][28], float weight[512][512][1][1]);

#endif
