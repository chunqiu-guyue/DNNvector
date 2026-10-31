__global__ void multiGradient_Add_l63(float* d_output0, float* d_output1, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        d_output0[idx] += d_output0[idx] + d_output1[idx];
      }
  }
}


__global__ void backwardPass_ReLU_l62(float d_input[64][512][28][28], float input[64][512][28][28], float d_output[64][512][28][28]) {

  long size = 64L*512*28*28;  //N * C * H * W;
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;
  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);  //n
    const long i2 = ((idx /= 64    ) % 512);  //c
    const long i3 = ((idx /= 512    ) % 28);  //h
    const long i4 = ((idx /= 28    ) % 28);  //w

    d_input[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? d_output[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void backwardPass_BatchNorm2d_l61(float d_output[64][512][28][28], float v1[64][512][28][28], float v2[512], float gamma_and_beta[1024], float dv1[64][512][28][28], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][28][28], float d_gamma_and_beta[1024]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 512L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[512 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*512*28*28;  // N * C * H * W
  const long N2 = 512L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*28*28;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l60(float input[64][256][28][28], float d_output[64][512][28][28], float d_weight[512][256][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 512L*256*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);   // k
    const long i2 = ((idx /= 512) % 256);   // c
    const long i3 = ((idx /= 256) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*256*512*28*28;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 256); //c
    const long i5 = ((idx /= 256  ) % 512); //k
    const long i6 = ((idx /= 512  ) % 28); //p
    const long i7 = ((idx /= 28  ) % 28); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l60(float d_input[64][256][28][28], float d_output[64][512][28][28], float weight[512][256][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*256*28*28;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 256);  //c
    const long i3 = ((idx /= 256   ) % 28);  //h
    const long i4 = ((idx /= 28   ) % 28);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*256*512*28*28;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 256);   //c
    const long i5 = ((idx /= 256    ) % 512);   //k
    const long i6 = ((idx /= 512    ) % 28);   //p
    const long i7 = ((idx /= 28    ) % 28);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l59(float d_output[64][256][28][28], float v1[64][256][28][28], float v2[256], float gamma_and_beta[512], float dv1[64][256][28][28], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][28][28], float d_gamma_and_beta[512]) { 
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

  const long N1 = 64L*256*28*28;  // N * C * H * W
  const long N2 = 256L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

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
  const long N_bn = 64L*28*28;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l58(float input[64][256][28][28], float d_output[64][256][28][28], float d_weight[256][256][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 256L*256*3*3;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);   // k
    const long i2 = ((idx /= 256) % 256);   // c
    const long i3 = ((idx /= 256) % 3);   // r
    const long i4 = ((idx /= 3) % 3);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*3*3*256*256*28*28;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 3); //r
    const long i3 = ((idx /= 3  ) % 3); //s
    const long i4 = ((idx /= 3  ) % 256); //c
    const long i5 = ((idx /= 256  ) % 256); //k
    const long i6 = ((idx /= 256  ) % 28); //p
    const long i7 = ((idx /= 28  ) % 28); //q

    const long h = i6 * 1 - 1 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 1 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l58(float d_input[64][256][28][28], float d_output[64][256][28][28], float weight[256][256][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*256*28*28;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 256);  //c
    const long i3 = ((idx /= 256   ) % 28);  //h
    const long i4 = ((idx /= 28   ) % 28);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*3*3*256*256*28*28;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 3);   //r
    const long i3 = ((idx /= 3    ) % 3);   //s
    const long i4 = ((idx /= 3    ) % 256);   //c
    const long i5 = ((idx /= 256    ) % 256);   //k
    const long i6 = ((idx /= 256    ) % 28);   //p
    const long i7 = ((idx /= 28    ) % 28);   //q

    const long h = i6 * 1 - 1 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 1 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l57(float d_output[64][256][28][28], float v1[64][256][28][28], float v2[256], float gamma_and_beta[512], float dv1[64][256][28][28], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][28][28], float d_gamma_and_beta[512]) { 
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

  const long N1 = 64L*256*28*28;  // N * C * H * W
  const long N2 = 256L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

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
  const long N_bn = 64L*28*28;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l56(float input[64][512][28][28], float d_output[64][256][28][28], float d_weight[256][512][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 256L*512*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);   // k
    const long i2 = ((idx /= 256) % 512);   // c
    const long i3 = ((idx /= 512) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*512*256*28*28;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 512); //c
    const long i5 = ((idx /= 512  ) % 256); //k
    const long i6 = ((idx /= 256  ) % 28); //p
    const long i7 = ((idx /= 28  ) % 28); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l56(float d_input[64][512][28][28], float d_output[64][256][28][28], float weight[256][512][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*512*28*28;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 512);  //c
    const long i3 = ((idx /= 512   ) % 28);  //h
    const long i4 = ((idx /= 28   ) % 28);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*512*256*28*28;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 512);   //c
    const long i5 = ((idx /= 512    ) % 256);   //k
    const long i6 = ((idx /= 256    ) % 28);   //p
    const long i7 = ((idx /= 28    ) % 28);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void multiGradient_Add_l55(float* d_output0, float* d_output1, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        d_output0[idx] += d_output0[idx] + d_output1[idx];
      }
  }
}


__global__ void backwardPass_ReLU_l54(float d_input[64][512][28][28], float input[64][512][28][28], float d_output[64][512][28][28]) {

  long size = 64L*512*28*28;  //N * C * H * W;
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;
  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);  //n
    const long i2 = ((idx /= 64    ) % 512);  //c
    const long i3 = ((idx /= 512    ) % 28);  //h
    const long i4 = ((idx /= 28    ) % 28);  //w

    d_input[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? d_output[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void backwardPass_BatchNorm2d_l53(float d_output[64][512][28][28], float v1[64][512][28][28], float v2[512], float gamma_and_beta[1024], float dv1[64][512][28][28], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][28][28], float d_gamma_and_beta[1024]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 512L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[512 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*512*28*28;  // N * C * H * W
  const long N2 = 512L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*28*28;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l52(float input[64][256][28][28], float d_output[64][512][28][28], float d_weight[512][256][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 512L*256*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);   // k
    const long i2 = ((idx /= 512) % 256);   // c
    const long i3 = ((idx /= 256) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*256*512*28*28;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 256); //c
    const long i5 = ((idx /= 256  ) % 512); //k
    const long i6 = ((idx /= 512  ) % 28); //p
    const long i7 = ((idx /= 28  ) % 28); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l52(float d_input[64][256][28][28], float d_output[64][512][28][28], float weight[512][256][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*256*28*28;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 256);  //c
    const long i3 = ((idx /= 256   ) % 28);  //h
    const long i4 = ((idx /= 28   ) % 28);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*256*512*28*28;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 256);   //c
    const long i5 = ((idx /= 256    ) % 512);   //k
    const long i6 = ((idx /= 512    ) % 28);   //p
    const long i7 = ((idx /= 28    ) % 28);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l51(float d_output[64][256][28][28], float v1[64][256][28][28], float v2[256], float gamma_and_beta[512], float dv1[64][256][28][28], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][28][28], float d_gamma_and_beta[512]) { 
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

  const long N1 = 64L*256*28*28;  // N * C * H * W
  const long N2 = 256L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

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
  const long N_bn = 64L*28*28;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l50(float input[64][256][28][28], float d_output[64][256][28][28], float d_weight[256][256][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 256L*256*3*3;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);   // k
    const long i2 = ((idx /= 256) % 256);   // c
    const long i3 = ((idx /= 256) % 3);   // r
    const long i4 = ((idx /= 3) % 3);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*3*3*256*256*28*28;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 3); //r
    const long i3 = ((idx /= 3  ) % 3); //s
    const long i4 = ((idx /= 3  ) % 256); //c
    const long i5 = ((idx /= 256  ) % 256); //k
    const long i6 = ((idx /= 256  ) % 28); //p
    const long i7 = ((idx /= 28  ) % 28); //q

    const long h = i6 * 1 - 1 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 1 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l50(float d_input[64][256][28][28], float d_output[64][256][28][28], float weight[256][256][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*256*28*28;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 256);  //c
    const long i3 = ((idx /= 256   ) % 28);  //h
    const long i4 = ((idx /= 28   ) % 28);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*3*3*256*256*28*28;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 3);   //r
    const long i3 = ((idx /= 3    ) % 3);   //s
    const long i4 = ((idx /= 3    ) % 256);   //c
    const long i5 = ((idx /= 256    ) % 256);   //k
    const long i6 = ((idx /= 256    ) % 28);   //p
    const long i7 = ((idx /= 28    ) % 28);   //q

    const long h = i6 * 1 - 1 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 1 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l49(float d_output[64][256][28][28], float v1[64][256][28][28], float v2[256], float gamma_and_beta[512], float dv1[64][256][28][28], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][28][28], float d_gamma_and_beta[512]) { 
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

  const long N1 = 64L*256*28*28;  // N * C * H * W
  const long N2 = 256L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

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
  const long N_bn = 64L*28*28;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l48(float input[64][512][28][28], float d_output[64][256][28][28], float d_weight[256][512][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 256L*512*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);   // k
    const long i2 = ((idx /= 256) % 512);   // c
    const long i3 = ((idx /= 512) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*512*256*28*28;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 512); //c
    const long i5 = ((idx /= 512  ) % 256); //k
    const long i6 = ((idx /= 256  ) % 28); //p
    const long i7 = ((idx /= 28  ) % 28); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l48(float d_input[64][512][28][28], float d_output[64][256][28][28], float weight[256][512][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*512*28*28;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 512);  //c
    const long i3 = ((idx /= 512   ) % 28);  //h
    const long i4 = ((idx /= 28   ) % 28);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*512*256*28*28;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 512);   //c
    const long i5 = ((idx /= 512    ) % 256);   //k
    const long i6 = ((idx /= 256    ) % 28);   //p
    const long i7 = ((idx /= 28    ) % 28);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void multiGradient_Add_l47(float* d_output0, float* d_output1, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        d_output0[idx] += d_output0[idx] + d_output1[idx];
      }
  }
}


__global__ void backwardPass_ReLU_l46(float d_input[64][512][28][28], float input[64][512][28][28], float d_output[64][512][28][28]) {

  long size = 64L*512*28*28;  //N * C * H * W;
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;
  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);  //n
    const long i2 = ((idx /= 64    ) % 512);  //c
    const long i3 = ((idx /= 512    ) % 28);  //h
    const long i4 = ((idx /= 28    ) % 28);  //w

    d_input[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? d_output[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void backwardPass_BatchNorm2d_l45(float d_output[64][512][28][28], float v1[64][512][28][28], float v2[512], float gamma_and_beta[1024], float dv1[64][512][28][28], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][28][28], float d_gamma_and_beta[1024]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 512L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[512 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*512*28*28;  // N * C * H * W
  const long N2 = 512L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*28*28;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l44(float input[64][256][28][28], float d_output[64][512][28][28], float d_weight[512][256][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 512L*256*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);   // k
    const long i2 = ((idx /= 512) % 256);   // c
    const long i3 = ((idx /= 256) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*256*512*28*28;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 256); //c
    const long i5 = ((idx /= 256  ) % 512); //k
    const long i6 = ((idx /= 512  ) % 28); //p
    const long i7 = ((idx /= 28  ) % 28); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l44(float d_input[64][256][28][28], float d_output[64][512][28][28], float weight[512][256][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*256*28*28;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 256);  //c
    const long i3 = ((idx /= 256   ) % 28);  //h
    const long i4 = ((idx /= 28   ) % 28);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*256*512*28*28;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 256);   //c
    const long i5 = ((idx /= 256    ) % 512);   //k
    const long i6 = ((idx /= 512    ) % 28);   //p
    const long i7 = ((idx /= 28    ) % 28);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l43(float d_output[64][256][28][28], float v1[64][256][28][28], float v2[256], float gamma_and_beta[512], float dv1[64][256][28][28], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][28][28], float d_gamma_and_beta[512]) { 
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

  const long N1 = 64L*256*28*28;  // N * C * H * W
  const long N2 = 256L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

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
  const long N_bn = 64L*28*28;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l42(float input[64][256][28][28], float d_output[64][256][28][28], float d_weight[256][256][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 256L*256*3*3;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);   // k
    const long i2 = ((idx /= 256) % 256);   // c
    const long i3 = ((idx /= 256) % 3);   // r
    const long i4 = ((idx /= 3) % 3);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*3*3*256*256*28*28;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 3); //r
    const long i3 = ((idx /= 3  ) % 3); //s
    const long i4 = ((idx /= 3  ) % 256); //c
    const long i5 = ((idx /= 256  ) % 256); //k
    const long i6 = ((idx /= 256  ) % 28); //p
    const long i7 = ((idx /= 28  ) % 28); //q

    const long h = i6 * 1 - 1 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 1 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l42(float d_input[64][256][28][28], float d_output[64][256][28][28], float weight[256][256][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*256*28*28;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 256);  //c
    const long i3 = ((idx /= 256   ) % 28);  //h
    const long i4 = ((idx /= 28   ) % 28);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*3*3*256*256*28*28;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 3);   //r
    const long i3 = ((idx /= 3    ) % 3);   //s
    const long i4 = ((idx /= 3    ) % 256);   //c
    const long i5 = ((idx /= 256    ) % 256);   //k
    const long i6 = ((idx /= 256    ) % 28);   //p
    const long i7 = ((idx /= 28    ) % 28);   //q

    const long h = i6 * 1 - 1 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 1 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l41(float d_output[64][256][28][28], float v1[64][256][28][28], float v2[256], float gamma_and_beta[512], float dv1[64][256][28][28], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][28][28], float d_gamma_and_beta[512]) { 
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

  const long N1 = 64L*256*28*28;  // N * C * H * W
  const long N2 = 256L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

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
  const long N_bn = 64L*28*28;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l40(float input[64][512][28][28], float d_output[64][256][28][28], float d_weight[256][512][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 256L*512*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);   // k
    const long i2 = ((idx /= 256) % 512);   // c
    const long i3 = ((idx /= 512) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*512*256*28*28;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 512); //c
    const long i5 = ((idx /= 512  ) % 256); //k
    const long i6 = ((idx /= 256  ) % 28); //p
    const long i7 = ((idx /= 28  ) % 28); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l40(float d_input[64][512][28][28], float d_output[64][256][28][28], float weight[256][512][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*512*28*28;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 512);  //c
    const long i3 = ((idx /= 512   ) % 28);  //h
    const long i4 = ((idx /= 28   ) % 28);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*512*256*28*28;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 512);   //c
    const long i5 = ((idx /= 512    ) % 256);   //k
    const long i6 = ((idx /= 256    ) % 28);   //p
    const long i7 = ((idx /= 28    ) % 28);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void multiGradient_Add_l39(float* d_output0, float* d_output1, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        d_output0[idx] += d_output0[idx] + d_output1[idx];
      }
  }
}


__global__ void backwardPass_BatchNorm2d_l38(float d_output[64][512][28][28], float v1[64][512][28][28], float v2[512], float gamma_and_beta[1024], float dv1[64][512][28][28], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][28][28], float d_gamma_and_beta[1024]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 512L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[512 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*512*28*28;  // N * C * H * W
  const long N2 = 512L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*28*28;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l37(float input[64][256][56][56], float d_output[64][512][28][28], float d_weight[512][256][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 512L*256*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);   // k
    const long i2 = ((idx /= 512) % 256);   // c
    const long i3 = ((idx /= 256) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*256*512*28*28;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 256); //c
    const long i5 = ((idx /= 256  ) % 512); //k
    const long i6 = ((idx /= 512  ) % 28); //p
    const long i7 = ((idx /= 28  ) % 28); //q

    const long h = i6 * 2 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 2 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l37(float d_input[64][256][56][56], float d_output[64][512][28][28], float weight[512][256][1][1])
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

  const long N = 64L*1*1*256*512*28*28;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 256);   //c
    const long i5 = ((idx /= 256    ) % 512);   //k
    const long i6 = ((idx /= 512    ) % 28);   //p
    const long i7 = ((idx /= 28    ) % 28);   //q

    const long h = i6 * 2 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 2 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_ReLU_l36(float d_input[64][512][28][28], float input[64][512][28][28], float d_output[64][512][28][28]) {

  long size = 64L*512*28*28;  //N * C * H * W;
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;
  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);  //n
    const long i2 = ((idx /= 64    ) % 512);  //c
    const long i3 = ((idx /= 512    ) % 28);  //h
    const long i4 = ((idx /= 28    ) % 28);  //w

    d_input[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? d_output[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void backwardPass_BatchNorm2d_l35(float d_output[64][512][28][28], float v1[64][512][28][28], float v2[512], float gamma_and_beta[1024], float dv1[64][512][28][28], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][28][28], float d_gamma_and_beta[1024]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 512L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[512 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*512*28*28;  // N * C * H * W
  const long N2 = 512L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*28*28;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l34(float input[64][256][28][28], float d_output[64][512][28][28], float d_weight[512][256][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 512L*256*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);   // k
    const long i2 = ((idx /= 512) % 256);   // c
    const long i3 = ((idx /= 256) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*256*512*28*28;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 256); //c
    const long i5 = ((idx /= 256  ) % 512); //k
    const long i6 = ((idx /= 512  ) % 28); //p
    const long i7 = ((idx /= 28  ) % 28); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l34(float d_input[64][256][28][28], float d_output[64][512][28][28], float weight[512][256][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*256*28*28;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 256);  //c
    const long i3 = ((idx /= 256   ) % 28);  //h
    const long i4 = ((idx /= 28   ) % 28);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*256*512*28*28;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 256);   //c
    const long i5 = ((idx /= 256    ) % 512);   //k
    const long i6 = ((idx /= 512    ) % 28);   //p
    const long i7 = ((idx /= 28    ) % 28);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l33(float d_output[64][256][28][28], float v1[64][256][28][28], float v2[256], float gamma_and_beta[512], float dv1[64][256][28][28], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][28][28], float d_gamma_and_beta[512]) { 
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

  const long N1 = 64L*256*28*28;  // N * C * H * W
  const long N2 = 256L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

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
  const long N_bn = 64L*28*28;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l32(float input[64][256][56][56], float d_output[64][256][28][28], float d_weight[256][256][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 256L*256*3*3;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);   // k
    const long i2 = ((idx /= 256) % 256);   // c
    const long i3 = ((idx /= 256) % 3);   // r
    const long i4 = ((idx /= 3) % 3);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*3*3*256*256*28*28;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 3); //r
    const long i3 = ((idx /= 3  ) % 3); //s
    const long i4 = ((idx /= 3  ) % 256); //c
    const long i5 = ((idx /= 256  ) % 256); //k
    const long i6 = ((idx /= 256  ) % 28); //p
    const long i7 = ((idx /= 28  ) % 28); //q

    const long h = i6 * 2 - 1 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 2 - 1 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l32(float d_input[64][256][56][56], float d_output[64][256][28][28], float weight[256][256][3][3])
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

  const long N = 64L*3*3*256*256*28*28;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 3);   //r
    const long i3 = ((idx /= 3    ) % 3);   //s
    const long i4 = ((idx /= 3    ) % 256);   //c
    const long i5 = ((idx /= 256    ) % 256);   //k
    const long i6 = ((idx /= 256    ) % 28);   //p
    const long i7 = ((idx /= 28    ) % 28);   //q

    const long h = i6 * 2 - 1 + i2; // h = p * stride - pad + r;
    const long w = i7 * 2 - 1 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 56 && w >= 0 && w < 56)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l31(float d_output[64][256][56][56], float v1[64][256][56][56], float v2[256], float gamma_and_beta[512], float dv1[64][256][56][56], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][56][56], float d_gamma_and_beta[512]) { 
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

__global__ void backwardPass_conv2d_Weight_l30(float input[64][256][56][56], float d_output[64][256][56][56], float d_weight[256][256][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 256L*256*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);   // k
    const long i2 = ((idx /= 256) % 256);   // c
    const long i3 = ((idx /= 256) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*256*256*56*56;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 256); //c
    const long i5 = ((idx /= 256  ) % 256); //k
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

__global__ void backwardPass_conv2d_Input_l30(float d_input[64][256][56][56], float d_output[64][256][56][56], float weight[256][256][1][1])
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

  const long N = 64L*1*1*256*256*56*56;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 256);   //c
    const long i5 = ((idx /= 256    ) % 256);   //k
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

__global__ void multiGradient_Add_l29(float* d_output0, float* d_output1, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        d_output0[idx] += d_output0[idx] + d_output1[idx];
      }
  }
}


__global__ void backwardPass_ReLU_l28(float d_input[64][256][56][56], float input[64][256][56][56], float d_output[64][256][56][56]) {

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

__global__ void backwardPass_BatchNorm2d_l27(float d_output[64][256][56][56], float v1[64][256][56][56], float v2[256], float gamma_and_beta[512], float dv1[64][256][56][56], float dv2[256], float dvar[256], float dmu[256], float d_input[64][256][56][56], float d_gamma_and_beta[512]) { 
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

