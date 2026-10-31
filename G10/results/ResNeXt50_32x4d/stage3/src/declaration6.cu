__global__ void backwardPass_conv2d_Weight_l26(float input[64][128][56][56], float d_output[64][256][56][56], float d_weight[256][128][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 256L*128*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);   // k
    const long i2 = ((idx /= 256) % 128);   // c
    const long i3 = ((idx /= 128) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*128*256*56*56;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 128); //c
    const long i5 = ((idx /= 128  ) % 256); //k
    const long i6 = ((idx /= 256  ) % 56); //p
    const long i7 = ((idx /= 56  ) % 56); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l26(float d_input[64][128][56][56], float d_output[64][256][56][56], float weight[256][128][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*128*56*56;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 128);  //c
    const long i3 = ((idx /= 128   ) % 56);  //h
    const long i4 = ((idx /= 56   ) % 56);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*128*256*56*56;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 128);   //c
    const long i5 = ((idx /= 128    ) % 256);   //k
    const long i6 = ((idx /= 256    ) % 56);   //p
    const long i7 = ((idx /= 56    ) % 56);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l25(float d_output[64][128][56][56], float v1[64][128][56][56], float v2[128], float gamma_and_beta[256], float dv1[64][128][56][56], float dv2[128], float dvar[128], float dmu[128], float d_input[64][128][56][56], float d_gamma_and_beta[256]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 128L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[128 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*128*56*56;  // N * C * H * W
  const long N2 = 128L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*56*56;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l24(float input[64][128][56][56], float d_output[64][128][56][56], float d_weight[128][128][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 128L*128*3*3;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);   // k
    const long i2 = ((idx /= 128) % 128);   // c
    const long i3 = ((idx /= 128) % 3);   // r
    const long i4 = ((idx /= 3) % 3);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*3*3*128*128*56*56;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 3); //r
    const long i3 = ((idx /= 3  ) % 3); //s
    const long i4 = ((idx /= 3  ) % 128); //c
    const long i5 = ((idx /= 128  ) % 128); //k
    const long i6 = ((idx /= 128  ) % 56); //p
    const long i7 = ((idx /= 56  ) % 56); //q

    const long h = i6 * 1 - 1 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 1 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l24(float d_input[64][128][56][56], float d_output[64][128][56][56], float weight[128][128][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*128*56*56;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 128);  //c
    const long i3 = ((idx /= 128   ) % 56);  //h
    const long i4 = ((idx /= 56   ) % 56);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*3*3*128*128*56*56;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 3);   //r
    const long i3 = ((idx /= 3    ) % 3);   //s
    const long i4 = ((idx /= 3    ) % 128);   //c
    const long i5 = ((idx /= 128    ) % 128);   //k
    const long i6 = ((idx /= 128    ) % 56);   //p
    const long i7 = ((idx /= 56    ) % 56);   //q

    const long h = i6 * 1 - 1 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 1 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l23(float d_output[64][128][56][56], float v1[64][128][56][56], float v2[128], float gamma_and_beta[256], float dv1[64][128][56][56], float dv2[128], float dvar[128], float dmu[128], float d_input[64][128][56][56], float d_gamma_and_beta[256]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 128L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[128 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*128*56*56;  // N * C * H * W
  const long N2 = 128L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*56*56;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l22(float input[64][256][56][56], float d_output[64][128][56][56], float d_weight[128][256][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 128L*256*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);   // k
    const long i2 = ((idx /= 128) % 256);   // c
    const long i3 = ((idx /= 256) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*256*128*56*56;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 256); //c
    const long i5 = ((idx /= 256  ) % 128); //k
    const long i6 = ((idx /= 128  ) % 56); //p
    const long i7 = ((idx /= 56  ) % 56); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l22(float d_input[64][256][56][56], float d_output[64][128][56][56], float weight[128][256][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*256*56*56;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 256);  //c
    const long i3 = ((idx /= 256   ) % 56);  //h
    const long i4 = ((idx /= 56   ) % 56);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*256*128*56*56;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 256);   //c
    const long i5 = ((idx /= 256    ) % 128);   //k
    const long i6 = ((idx /= 128    ) % 56);   //p
    const long i7 = ((idx /= 56    ) % 56);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void multiGradient_Add_l21(float* d_output0, float* d_output1, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        d_output0[idx] += d_output0[idx] + d_output1[idx];
      }
  }
}


__global__ void backwardPass_ReLU_l20(float d_input[64][256][56][56], float input[64][256][56][56], float d_output[64][256][56][56]) {

  long size = 64L*256*56*56;  //N * C * H * W;
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;
  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);  //n
    const long i2 = ((idx /= 64    ) % 256);  //c
    const long i3 = ((idx /= 256    ) % 56);  //h
    const long i4 = ((idx /= 56    ) % 56);  //w

    d_input[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? d_output[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void backwardPass_BatchNorm2d_l19(float d_output[64][256][56][56], float v1[64][256][56][56], float v2[256], float gamma_and_beta[512], float dv1[64][256][56][56], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][56][56], float d_gamma_and_beta[512]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 256L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[256 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*256*56*56;  // N * C * H * W
  const long N2 = 256L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*56*56;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l18(float input[64][128][56][56], float d_output[64][256][56][56], float d_weight[256][128][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 256L*128*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);   // k
    const long i2 = ((idx /= 256) % 128);   // c
    const long i3 = ((idx /= 128) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*128*256*56*56;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 128); //c
    const long i5 = ((idx /= 128  ) % 256); //k
    const long i6 = ((idx /= 256  ) % 56); //p
    const long i7 = ((idx /= 56  ) % 56); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l18(float d_input[64][128][56][56], float d_output[64][256][56][56], float weight[256][128][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*128*56*56;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 128);  //c
    const long i3 = ((idx /= 128   ) % 56);  //h
    const long i4 = ((idx /= 56   ) % 56);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*128*256*56*56;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 128);   //c
    const long i5 = ((idx /= 128    ) % 256);   //k
    const long i6 = ((idx /= 256    ) % 56);   //p
    const long i7 = ((idx /= 56    ) % 56);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l17(float d_output[64][128][56][56], float v1[64][128][56][56], float v2[128], float gamma_and_beta[256], float dv1[64][128][56][56], float dv2[128], float dvar[128], float dmu[128], float d_input[64][128][56][56], float d_gamma_and_beta[256]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 128L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[128 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*128*56*56;  // N * C * H * W
  const long N2 = 128L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*56*56;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l16(float input[64][128][56][56], float d_output[64][128][56][56], float d_weight[128][128][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 128L*128*3*3;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);   // k
    const long i2 = ((idx /= 128) % 128);   // c
    const long i3 = ((idx /= 128) % 3);   // r
    const long i4 = ((idx /= 3) % 3);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*3*3*128*128*56*56;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 3); //r
    const long i3 = ((idx /= 3  ) % 3); //s
    const long i4 = ((idx /= 3  ) % 128); //c
    const long i5 = ((idx /= 128  ) % 128); //k
    const long i6 = ((idx /= 128  ) % 56); //p
    const long i7 = ((idx /= 56  ) % 56); //q

    const long h = i6 * 1 - 1 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 1 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l16(float d_input[64][128][56][56], float d_output[64][128][56][56], float weight[128][128][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*128*56*56;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 128);  //c
    const long i3 = ((idx /= 128   ) % 56);  //h
    const long i4 = ((idx /= 56   ) % 56);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*3*3*128*128*56*56;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 3);   //r
    const long i3 = ((idx /= 3    ) % 3);   //s
    const long i4 = ((idx /= 3    ) % 128);   //c
    const long i5 = ((idx /= 128    ) % 128);   //k
    const long i6 = ((idx /= 128    ) % 56);   //p
    const long i7 = ((idx /= 56    ) % 56);   //q

    const long h = i6 * 1 - 1 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 1 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l15(float d_output[64][128][56][56], float v1[64][128][56][56], float v2[128], float gamma_and_beta[256], float dv1[64][128][56][56], float dv2[128], float dvar[128], float dmu[128], float d_input[64][128][56][56], float d_gamma_and_beta[256]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 128L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[128 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*128*56*56;  // N * C * H * W
  const long N2 = 128L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*56*56;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l14(float input[64][256][56][56], float d_output[64][128][56][56], float d_weight[128][256][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 128L*256*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);   // k
    const long i2 = ((idx /= 128) % 256);   // c
    const long i3 = ((idx /= 256) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*256*128*56*56;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 256); //c
    const long i5 = ((idx /= 256  ) % 128); //k
    const long i6 = ((idx /= 128  ) % 56); //p
    const long i7 = ((idx /= 56  ) % 56); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l14(float d_input[64][256][56][56], float d_output[64][128][56][56], float weight[128][256][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*256*56*56;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 256);  //c
    const long i3 = ((idx /= 256   ) % 56);  //h
    const long i4 = ((idx /= 56   ) % 56);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*256*128*56*56;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 256);   //c
    const long i5 = ((idx /= 256    ) % 128);   //k
    const long i6 = ((idx /= 128    ) % 56);   //p
    const long i7 = ((idx /= 56    ) % 56);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void multiGradient_Add_l13(float* d_output0, float* d_output1, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        d_output0[idx] += d_output0[idx] + d_output1[idx];
      }
  }
}


__global__ void backwardPass_BatchNorm2d_l12(float d_output[64][256][56][56], float v1[64][256][56][56], float v2[256], float gamma_and_beta[512], float dv1[64][256][56][56], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][56][56], float d_gamma_and_beta[512]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 256L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[256 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*256*56*56;  // N * C * H * W
  const long N2 = 256L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*56*56;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l11(float input[64][64][56][56], float d_output[64][256][56][56], float d_weight[256][64][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 256L*64*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);   // k
    const long i2 = ((idx /= 256) % 64);   // c
    const long i3 = ((idx /= 64) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*64*256*56*56;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 64); //c
    const long i5 = ((idx /= 64  ) % 256); //k
    const long i6 = ((idx /= 256  ) % 56); //p
    const long i7 = ((idx /= 56  ) % 56); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l11(float d_input[64][64][56][56], float d_output[64][256][56][56], float weight[256][64][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*64*56*56;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 64);  //c
    const long i3 = ((idx /= 64   ) % 56);  //h
    const long i4 = ((idx /= 56   ) % 56);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*64*256*56*56;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 64);   //c
    const long i5 = ((idx /= 64    ) % 256);   //k
    const long i6 = ((idx /= 256    ) % 56);   //p
    const long i7 = ((idx /= 56    ) % 56);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_ReLU_l10(float d_input[64][256][56][56], float input[64][256][56][56], float d_output[64][256][56][56]) {

  long size = 64L*256*56*56;  //N * C * H * W;
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;
  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);  //n
    const long i2 = ((idx /= 64    ) % 256);  //c
    const long i3 = ((idx /= 256    ) % 56);  //h
    const long i4 = ((idx /= 56    ) % 56);  //w

    d_input[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? d_output[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void backwardPass_BatchNorm2d_l9(float d_output[64][256][56][56], float v1[64][256][56][56], float v2[256], float gamma_and_beta[512], float dv1[64][256][56][56], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][56][56], float d_gamma_and_beta[512]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 256L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[256 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*256*56*56;  // N * C * H * W
  const long N2 = 256L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*56*56;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l8(float input[64][128][56][56], float d_output[64][256][56][56], float d_weight[256][128][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 256L*128*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);   // k
    const long i2 = ((idx /= 256) % 128);   // c
    const long i3 = ((idx /= 128) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*128*256*56*56;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 128); //c
    const long i5 = ((idx /= 128  ) % 256); //k
    const long i6 = ((idx /= 256  ) % 56); //p
    const long i7 = ((idx /= 56  ) % 56); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l8(float d_input[64][128][56][56], float d_output[64][256][56][56], float weight[256][128][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*128*56*56;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 128);  //c
    const long i3 = ((idx /= 128   ) % 56);  //h
    const long i4 = ((idx /= 56   ) % 56);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*128*256*56*56;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 128);   //c
    const long i5 = ((idx /= 128    ) % 256);   //k
    const long i6 = ((idx /= 256    ) % 56);   //p
    const long i7 = ((idx /= 56    ) % 56);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l7(float d_output[64][128][56][56], float v1[64][128][56][56], float v2[128], float gamma_and_beta[256], float dv1[64][128][56][56], float dv2[128], float dvar[128], float dmu[128], float d_input[64][128][56][56], float d_gamma_and_beta[256]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 128L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[128 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*128*56*56;  // N * C * H * W
  const long N2 = 128L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*56*56;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l6(float input[64][128][56][56], float d_output[64][128][56][56], float d_weight[128][128][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 128L*128*3*3;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);   // k
    const long i2 = ((idx /= 128) % 128);   // c
    const long i3 = ((idx /= 128) % 3);   // r
    const long i4 = ((idx /= 3) % 3);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*3*3*128*128*56*56;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 3); //r
    const long i3 = ((idx /= 3  ) % 3); //s
    const long i4 = ((idx /= 3  ) % 128); //c
    const long i5 = ((idx /= 128  ) % 128); //k
    const long i6 = ((idx /= 128  ) % 56); //p
    const long i7 = ((idx /= 56  ) % 56); //q

    const long h = i6 * 1 - 1 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 1 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l6(float d_input[64][128][56][56], float d_output[64][128][56][56], float weight[128][128][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*128*56*56;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 128);  //c
    const long i3 = ((idx /= 128   ) % 56);  //h
    const long i4 = ((idx /= 56   ) % 56);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*3*3*128*128*56*56;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 3);   //r
    const long i3 = ((idx /= 3    ) % 3);   //s
    const long i4 = ((idx /= 3    ) % 128);   //c
    const long i5 = ((idx /= 128    ) % 128);   //k
    const long i6 = ((idx /= 128    ) % 56);   //p
    const long i7 = ((idx /= 56    ) % 56);   //q

    const long h = i6 * 1 - 1 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 1 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l5(float d_output[64][128][56][56], float v1[64][128][56][56], float v2[128], float gamma_and_beta[256], float dv1[64][128][56][56], float dv2[128], float dvar[128], float dmu[128], float d_input[64][128][56][56], float d_gamma_and_beta[256]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 128L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[128 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*128*56*56;  // N * C * H * W
  const long N2 = 128L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*56*56;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l4(float input[64][64][56][56], float d_output[64][128][56][56], float d_weight[128][64][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 128L*64*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);   // k
    const long i2 = ((idx /= 128) % 64);   // c
    const long i3 = ((idx /= 64) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*64*128*56*56;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 64); //c
    const long i5 = ((idx /= 64  ) % 128); //k
    const long i6 = ((idx /= 128  ) % 56); //p
    const long i7 = ((idx /= 56  ) % 56); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l4(float d_input[64][64][56][56], float d_output[64][128][56][56], float weight[128][64][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*64*56*56;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 64);  //c
    const long i3 = ((idx /= 64   ) % 56);  //h
    const long i4 = ((idx /= 56   ) % 56);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*64*128*56*56;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 64);   //c
    const long i5 = ((idx /= 64    ) % 128);   //k
    const long i6 = ((idx /= 128    ) % 56);   //p
    const long i7 = ((idx /= 56    ) % 56);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void multiGradient_Add_l3(float* d_output0, float* d_output1, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        d_output0[idx] += d_output0[idx] + d_output1[idx];
      }
  }
}


__global__ void backwardPass_MaxPool2d_l3(float d_input[64][64][112][112], float input[64][64][112][112], float d_output[64][64][56][56])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*64*112*112;  // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 64);  //c
    const long i3 = ((idx /= 64   ) % 112);  //h
    const long i4 = ((idx /= 112   ) % 112);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*64*56*56;   // N * K * P * Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64);   //n
    const long i2 = ((idx /= 64  ) % 64);   //c
    const long i3 = ((idx /= 64  ) % 56);   //p
    const long i4 = ((idx /= 56  ) % 56);   //q

    long max_idx_k0 = 0;
    long max_idx_k1 = 0;
    float max = -9999999;

    for (long k0 = 0; k0 < 3; k0++)  //kernelsize_0
    {
      for (long k1 = 0; k1 < 3; k1++)  // kernelsize_1
      {
        const long h = i3 * 2 - 1 + k0; // h = p * stride - pad + k0;
        const long w = i4 * 2 - 1 + k1; // w = q * stride - pad + k1;
        if (h >= 0 && h < 112 && w >= 0 && w < 112)
        {
          if (max = max < input[i1][i2][h][w])
          {
            max = input[i1][i2][h][w];
            max_idx_k0 = k0;
            max_idx_k1 = k1;      //Finding where the max value is
          }
        }
      }
    }

    const long h_max = i3 * 2 - 1 + max_idx_k0;   // h = p * stride - pad + k0;
    const long w_max = i4 * 2 - 1 + max_idx_k1;   // w = p * stride - pad + k1;

    atomicAdd(&d_input[i1][i2][h_max][w_max], d_output[i1][i2][i3][i4]);
  }
}

__global__ void backwardPass_ReLU_l2(float d_input[64][64][112][112], float input[64][64][112][112], float d_output[64][64][112][112]) {

  long size = 64L*64*112*112;  //N * C * H * W;
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;
  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);  //n
    const long i2 = ((idx /= 64    ) % 64);  //c
    const long i3 = ((idx /= 64    ) % 112);  //h
    const long i4 = ((idx /= 112    ) % 112);  //w

    d_input[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? d_output[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void backwardPass_BatchNorm2d_l1(float d_output[64][64][112][112], float v1[64][64][112][112], float v2[64], float gamma_and_beta[128], float dv1[64][64][112][112], float dv2[64], float dvar[64], float dmu[64], float d_input[64][64][112][112], float d_gamma_and_beta[128]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 64L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[64 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*64*112*112;  // N * C * H * W
  const long N2 = 64L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 64); // c
    const long i3 = ((idx /= 64) % 112); // h
    const long i4 = ((idx /= 112) % 112); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 64); // c
    const long i3 = ((idx /= 64) % 112); // h
    const long i4 = ((idx /= 112) % 112); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*112*112;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 64); // c
    const long i3 = ((idx /= 64) % 112); // h
    const long i4 = ((idx /= 112) % 112); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 64); // c
    const long i3 = ((idx /= 64) % 112); // h
    const long i4 = ((idx /= 112) % 112); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 64); // c
    const long i3 = ((idx /= 64) % 112); // h
    const long i4 = ((idx /= 112) % 112); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l0(float input[64][3][224][224], float d_output[64][64][112][112], float d_weight[64][3][7][7])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*3*7*7;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);   // k
    const long i2 = ((idx /= 64) % 3);   // c
    const long i3 = ((idx /= 3) % 7);   // r
    const long i4 = ((idx /= 7) % 7);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*7*7*3*64*112*112;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 7); //r
    const long i3 = ((idx /= 7  ) % 7); //s
    const long i4 = ((idx /= 7  ) % 3); //c
    const long i5 = ((idx /= 3  ) % 64); //k
    const long i6 = ((idx /= 64  ) % 112); //p
    const long i7 = ((idx /= 112  ) % 112); //q

    const long h = i6 * 2 - 3 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 2 - 3 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 224 && w >= 0 && w < 224)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l0(float d_input[64][3][224][224], float d_output[64][64][112][112], float weight[64][3][7][7])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*3*224*224;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 3);  //c
    const long i3 = ((idx /= 3   ) % 224);  //h
    const long i4 = ((idx /= 224   ) % 224);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*7*7*3*64*112*112;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 7);   //r
    const long i3 = ((idx /= 7    ) % 7);   //s
    const long i4 = ((idx /= 7    ) % 3);   //c
    const long i5 = ((idx /= 3    ) % 64);   //k
    const long i6 = ((idx /= 64    ) % 112);   //p
    const long i7 = ((idx /= 112    ) % 112);   //q

    const long h = i6 * 2 - 3 + i2; // h = p * stride - pad + r;
    const long w = i7 * 2 - 3 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 224 && w >= 0 && w < 224)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

