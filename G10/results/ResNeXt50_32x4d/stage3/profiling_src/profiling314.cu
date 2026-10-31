#include "../include/declaration5.cuh"
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
float *tensor430, *tensor434, *tensor435, *tensor448, *tensor436, *tensor437, *tensor446, *tensor447;
float *tensor438, *tensor439, *tensor440, *tensor441, *tensor442, *tensor443, *tensor444, *tensor445;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][512][28][28], V1:[64][512][28][28], V2:[512], Gamma_beta:[1024], D_v1:[64][512][28][28], D_v2:[512], D_var:[512], D_mu:[512], D_input:[64][512][28][28], D_gamma_beta:[1024]
  backwardPass_BatchNorm2d_l61<<<64, 0>>>((float (*)[512][28][28]) tensor448, (float (*)[512][28][28]) tensor440, (float (*)) tensor441, (float (*)) tensor436, (float (*)[512][28][28]) tensor444, (float (*)) tensor445, (float (*)) tensor443, (float (*)) tensor442, (float (*)[512][28][28]) tensor435, (float (*)) tensor437);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("314 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 616603648);
  
  // Weight addr assignment
  tensor430 = workspace_base + 0;
  tensor434 = workspace_base + 25690112;
  tensor435 = workspace_base + 51380224;
  tensor448 = workspace_base + 77070336;
  tensor436 = workspace_base + 102760448;
  tensor437 = workspace_base + 102761472;
  tensor446 = workspace_base + 102762496;
  tensor447 = workspace_base + 102763520;
  tensor438 = workspace_base + 102764544;
  tensor439 = workspace_base + 102765568;
  tensor440 = workspace_base + 102766592;
  tensor441 = workspace_base + 128456704;
  tensor442 = workspace_base + 128457728;
  tensor443 = workspace_base + 128458752;
  tensor444 = workspace_base + 128459776;
  tensor445 = workspace_base + 154149888;

  // Data & mask generation
  generate_data(tensor430, 25690112);
  generate_data(tensor434, 25690112);
  generate_data(tensor435, 25690112);
  generate_data(tensor448, 25690112);
  generate_data(tensor436, 1024);
  generate_data(tensor437, 1024);
  generate_data(tensor446, 1024);
  generate_data(tensor447, 1024);
  generate_data(tensor438, 1024);
  generate_data(tensor439, 1024);
  generate_data(tensor440, 25690112);
  generate_data(tensor441, 1024);
  generate_data(tensor442, 1024);
  generate_data(tensor443, 1024);
  generate_data(tensor444, 25690112);
  generate_data(tensor445, 1024);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
