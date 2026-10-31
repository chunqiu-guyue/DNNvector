#include "../include/declaration3.cuh"
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
float *tensor840, *tensor844, *tensor845, *tensor858, *tensor846, *tensor847, *tensor856, *tensor857;
float *tensor848, *tensor849, *tensor850, *tensor851, *tensor852, *tensor853, *tensor854, *tensor855;

void run_iteration() {
  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  // D_out:[64][2048][7][7], V1:[64][2048][7][7], V2:[2048], Gamma_beta:[4096], D_v1:[64][2048][7][7], D_v2:[2048], D_var:[2048], D_mu:[2048], D_input:[64][2048][7][7], D_gamma_beta:[4096]
  backwardPass_BatchNorm2d_l119<<<64, 0>>>((float (*)[2048][7][7]) tensor858, (float (*)[2048][7][7]) tensor850, (float (*)) tensor851, (float (*)) tensor846, (float (*)[2048][7][7]) tensor854, (float (*)) tensor855, (float (*)) tensor853, (float (*)) tensor852, (float (*)[2048][7][7]) tensor845, (float (*)) tensor847);

  cudaEventRecord(stop, 0);
  CUDA_CHECK_RET( cudaEventSynchronize(stop) );
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("190 %20.10f ms\n", elapsedTime);
}

// main function
int main(int argc, const  char **argv) {
  // Workspace malloc
  cudaMallocManaged(&workspace_base, 154238976);
  
  // Weight addr assignment
  tensor840 = workspace_base + 0;
  tensor844 = workspace_base + 6422528;
  tensor845 = workspace_base + 12845056;
  tensor858 = workspace_base + 19267584;
  tensor846 = workspace_base + 25690112;
  tensor847 = workspace_base + 25694208;
  tensor856 = workspace_base + 25698304;
  tensor857 = workspace_base + 25700352;
  tensor848 = workspace_base + 25702400;
  tensor849 = workspace_base + 25704448;
  tensor850 = workspace_base + 25706496;
  tensor851 = workspace_base + 32129024;
  tensor852 = workspace_base + 32131072;
  tensor853 = workspace_base + 32133120;
  tensor854 = workspace_base + 32135168;
  tensor855 = workspace_base + 38557696;

  // Data & mask generation
  generate_data(tensor840, 6422528);
  generate_data(tensor844, 6422528);
  generate_data(tensor845, 6422528);
  generate_data(tensor858, 6422528);
  generate_data(tensor846, 4096);
  generate_data(tensor847, 4096);
  generate_data(tensor856, 2048);
  generate_data(tensor857, 2048);
  generate_data(tensor848, 2048);
  generate_data(tensor849, 2048);
  generate_data(tensor850, 6422528);
  generate_data(tensor851, 2048);
  generate_data(tensor852, 2048);
  generate_data(tensor853, 2048);
  generate_data(tensor854, 6422528);
  generate_data(tensor855, 2048);

  // Run iteration
  run_iteration();

  // Free
  cudaFree(workspace_base);
}
