#include "../include/declaration6.cuh"
#include <cuda.h>
#include <cstdio>
#include <stdlib.h>
#include <time.h>

// Error checking marco
#define CUDA_CHECK_RET(x)                                               \
  do {                                                                  \
    cudaError_t err_code = x;                                           \
    if (err_code != 0) {                                                \
      printf("Failed at %s:%d w/ code %d <%s>, abort\n",                \
          __FILE__, __LINE__, err_code, cudaGetErrorString(err_code));  \
      exit(1);                                                          \
    }                                                                   \
  } while (0)

// Workspace
float *workspace_base;

// helper functions
void generate_data(float *base, long long size) {
  for(long long i = 0; i < size; i++){
    base[i] = (float)(rand() % 256); 
  }
}

void generate_mask(float *base, long long size, float p) {
  for(long long i = 0; i < size; i++){
    float m = (float) (rand() % 1000000); 
    base[i] = m < (p * 1000000) ? 1.0f : 0.0f;
  }
}

// declare all tensors
float *tensor22, *tensor26, *tensor27, *tensor43, *tensor28, *tensor29, *tensor38, *tensor39;
float *tensor30, *tensor31, *tensor32, *tensor33, *tensor34, *tensor35, *tensor36, *tensor37;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][128][56][56], V1:[64][128][56][56], V2:[128], Gamma_beta:[256], D_v1:[64][128][56][56], D_v2:[128], D_var:[128], D_mu:[128], D_input:[64][128][56][56], D_gamma_beta:[256]
  backwardPass_BatchNorm2d_l5<<<64, 0>>>((float (*)[128][56][56]) tensor43, (float (*)[128][56][56]) tensor32, (float (*)) tensor33, (float (*)) tensor28, (float (*)[128][56][56]) tensor36, (float (*)) tensor37, (float (*)) tensor35, (float (*)) tensor34, (float (*)[128][56][56]) tensor27, (float (*)) tensor29);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("436 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 616603648);
  
  // Weight addr assignment
  tensor22 = workspace_base + 0;
  tensor26 = workspace_base + 25690112;
  tensor27 = workspace_base + 51380224;
  tensor43 = workspace_base + 77070336;
  tensor28 = workspace_base + 102760448;
  tensor29 = workspace_base + 102761472;
  tensor38 = workspace_base + 102762496;
  tensor39 = workspace_base + 102763520;
  tensor30 = workspace_base + 102764544;
  tensor31 = workspace_base + 102765568;
  tensor32 = workspace_base + 102766592;
  tensor33 = workspace_base + 128456704;
  tensor34 = workspace_base + 128457728;
  tensor35 = workspace_base + 128458752;
  tensor36 = workspace_base + 128459776;
  tensor37 = workspace_base + 154149888;

  // Data & mask generation
  generate_data(tensor22, 25690112);
  generate_data(tensor26, 25690112);
  generate_data(tensor27, 25690112);
  generate_data(tensor43, 25690112);
  generate_data(tensor28, 1024);
  generate_data(tensor29, 1024);
  generate_data(tensor38, 1024);
  generate_data(tensor39, 1024);
  generate_data(tensor30, 1024);
  generate_data(tensor31, 1024);
  generate_data(tensor32, 25690112);
  generate_data(tensor33, 1024);
  generate_data(tensor34, 1024);
  generate_data(tensor35, 1024);
  generate_data(tensor36, 25690112);
  generate_data(tensor37, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
