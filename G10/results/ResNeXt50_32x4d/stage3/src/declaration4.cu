__global__ void backwardPass_BatchNorm2d_l99(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]) { 
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

  const long N1 = 64L*512*14*14;  // N * C * H * W
  const long N2 = 512L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

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
  const long N_bn = 64L*14*14;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l98(float input[64][1024][14][14], float d_output[64][512][14][14], float d_weight[512][1024][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 512L*1024*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);   // k
    const long i2 = ((idx /= 512) % 1024);   // c
    const long i3 = ((idx /= 1024) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*1024*512*14*14;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 1024); //c
    const long i5 = ((idx /= 1024  ) % 512); //k
    const long i6 = ((idx /= 512  ) % 14); //p
    const long i7 = ((idx /= 14  ) % 14); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l98(float d_input[64][1024][14][14], float d_output[64][512][14][14], float weight[512][1024][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*1024*14*14;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 1024);  //c
    const long i3 = ((idx /= 1024   ) % 14);  //h
    const long i4 = ((idx /= 14   ) % 14);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*1024*512*14*14;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 1024);   //c
    const long i5 = ((idx /= 1024    ) % 512);   //k
    const long i6 = ((idx /= 512    ) % 14);   //p
    const long i7 = ((idx /= 14    ) % 14);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void multiGradient_Add_l97(float* d_output0, float* d_output1, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        d_output0[idx] += d_output0[idx] + d_output1[idx];
      }
  }
}


__global__ void backwardPass_ReLU_l96(float d_input[64][1024][14][14], float input[64][1024][14][14], float d_output[64][1024][14][14]) {

  long size = 64L*1024*14*14;  //N * C * H * W;
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;
  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);  //n
    const long i2 = ((idx /= 64    ) % 1024);  //c
    const long i3 = ((idx /= 1024    ) % 14);  //h
    const long i4 = ((idx /= 14    ) % 14);  //w

    d_input[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? d_output[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void backwardPass_BatchNorm2d_l95(float d_output[64][1024][14][14], float v1[64][1024][14][14], float v2[1024], float gamma_and_beta[2048], float dv1[64][1024][14][14], float dv2[1024], float dvar[1024], float dmu[1024], float d_input[64][1024][14][14], float d_gamma_and_beta[2048]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 1024L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[1024 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*1024*14*14;  // N * C * H * W
  const long N2 = 1024L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*14*14;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l94(float input[64][512][14][14], float d_output[64][1024][14][14], float d_weight[1024][512][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 1024L*512*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);   // k
    const long i2 = ((idx /= 1024) % 512);   // c
    const long i3 = ((idx /= 512) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*512*1024*14*14;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 512); //c
    const long i5 = ((idx /= 512  ) % 1024); //k
    const long i6 = ((idx /= 1024  ) % 14); //p
    const long i7 = ((idx /= 14  ) % 14); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l94(float d_input[64][512][14][14], float d_output[64][1024][14][14], float weight[1024][512][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*512*14*14;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 512);  //c
    const long i3 = ((idx /= 512   ) % 14);  //h
    const long i4 = ((idx /= 14   ) % 14);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*512*1024*14*14;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 512);   //c
    const long i5 = ((idx /= 512    ) % 1024);   //k
    const long i6 = ((idx /= 1024    ) % 14);   //p
    const long i7 = ((idx /= 14    ) % 14);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l93(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]) { 
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

  const long N1 = 64L*512*14*14;  // N * C * H * W
  const long N2 = 512L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

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
  const long N_bn = 64L*14*14;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l92(float input[64][512][14][14], float d_output[64][512][14][14], float d_weight[512][512][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 512L*512*3*3;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);   // k
    const long i2 = ((idx /= 512) % 512);   // c
    const long i3 = ((idx /= 512) % 3);   // r
    const long i4 = ((idx /= 3) % 3);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*3*3*512*512*14*14;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 3); //r
    const long i3 = ((idx /= 3  ) % 3); //s
    const long i4 = ((idx /= 3  ) % 512); //c
    const long i5 = ((idx /= 512  ) % 512); //k
    const long i6 = ((idx /= 512  ) % 14); //p
    const long i7 = ((idx /= 14  ) % 14); //q

    const long h = i6 * 1 - 1 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 1 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l92(float d_input[64][512][14][14], float d_output[64][512][14][14], float weight[512][512][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*512*14*14;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 512);  //c
    const long i3 = ((idx /= 512   ) % 14);  //h
    const long i4 = ((idx /= 14   ) % 14);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*3*3*512*512*14*14;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 3);   //r
    const long i3 = ((idx /= 3    ) % 3);   //s
    const long i4 = ((idx /= 3    ) % 512);   //c
    const long i5 = ((idx /= 512    ) % 512);   //k
    const long i6 = ((idx /= 512    ) % 14);   //p
    const long i7 = ((idx /= 14    ) % 14);   //q

    const long h = i6 * 1 - 1 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 1 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l91(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]) { 
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

  const long N1 = 64L*512*14*14;  // N * C * H * W
  const long N2 = 512L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

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
  const long N_bn = 64L*14*14;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l90(float input[64][1024][14][14], float d_output[64][512][14][14], float d_weight[512][1024][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 512L*1024*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);   // k
    const long i2 = ((idx /= 512) % 1024);   // c
    const long i3 = ((idx /= 1024) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*1024*512*14*14;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 1024); //c
    const long i5 = ((idx /= 1024  ) % 512); //k
    const long i6 = ((idx /= 512  ) % 14); //p
    const long i7 = ((idx /= 14  ) % 14); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l90(float d_input[64][1024][14][14], float d_output[64][512][14][14], float weight[512][1024][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*1024*14*14;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 1024);  //c
    const long i3 = ((idx /= 1024   ) % 14);  //h
    const long i4 = ((idx /= 14   ) % 14);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*1024*512*14*14;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 1024);   //c
    const long i5 = ((idx /= 1024    ) % 512);   //k
    const long i6 = ((idx /= 512    ) % 14);   //p
    const long i7 = ((idx /= 14    ) % 14);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void multiGradient_Add_l89(float* d_output0, float* d_output1, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        d_output0[idx] += d_output0[idx] + d_output1[idx];
      }
  }
}


__global__ void backwardPass_ReLU_l88(float d_input[64][1024][14][14], float input[64][1024][14][14], float d_output[64][1024][14][14]) {

  long size = 64L*1024*14*14;  //N * C * H * W;
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;
  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);  //n
    const long i2 = ((idx /= 64    ) % 1024);  //c
    const long i3 = ((idx /= 1024    ) % 14);  //h
    const long i4 = ((idx /= 14    ) % 14);  //w

    d_input[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? d_output[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void backwardPass_BatchNorm2d_l87(float d_output[64][1024][14][14], float v1[64][1024][14][14], float v2[1024], float gamma_and_beta[2048], float dv1[64][1024][14][14], float dv2[1024], float dvar[1024], float dmu[1024], float d_input[64][1024][14][14], float d_gamma_and_beta[2048]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 1024L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[1024 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*1024*14*14;  // N * C * H * W
  const long N2 = 1024L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*14*14;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l86(float input[64][512][14][14], float d_output[64][1024][14][14], float d_weight[1024][512][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 1024L*512*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);   // k
    const long i2 = ((idx /= 1024) % 512);   // c
    const long i3 = ((idx /= 512) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*512*1024*14*14;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 512); //c
    const long i5 = ((idx /= 512  ) % 1024); //k
    const long i6 = ((idx /= 1024  ) % 14); //p
    const long i7 = ((idx /= 14  ) % 14); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l86(float d_input[64][512][14][14], float d_output[64][1024][14][14], float weight[1024][512][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*512*14*14;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 512);  //c
    const long i3 = ((idx /= 512   ) % 14);  //h
    const long i4 = ((idx /= 14   ) % 14);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*512*1024*14*14;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 512);   //c
    const long i5 = ((idx /= 512    ) % 1024);   //k
    const long i6 = ((idx /= 1024    ) % 14);   //p
    const long i7 = ((idx /= 14    ) % 14);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l85(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]) { 
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

  const long N1 = 64L*512*14*14;  // N * C * H * W
  const long N2 = 512L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

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
  const long N_bn = 64L*14*14;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l84(float input[64][512][14][14], float d_output[64][512][14][14], float d_weight[512][512][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 512L*512*3*3;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);   // k
    const long i2 = ((idx /= 512) % 512);   // c
    const long i3 = ((idx /= 512) % 3);   // r
    const long i4 = ((idx /= 3) % 3);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*3*3*512*512*14*14;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 3); //r
    const long i3 = ((idx /= 3  ) % 3); //s
    const long i4 = ((idx /= 3  ) % 512); //c
    const long i5 = ((idx /= 512  ) % 512); //k
    const long i6 = ((idx /= 512  ) % 14); //p
    const long i7 = ((idx /= 14  ) % 14); //q

    const long h = i6 * 1 - 1 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 1 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l84(float d_input[64][512][14][14], float d_output[64][512][14][14], float weight[512][512][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*512*14*14;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 512);  //c
    const long i3 = ((idx /= 512   ) % 14);  //h
    const long i4 = ((idx /= 14   ) % 14);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*3*3*512*512*14*14;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 3);   //r
    const long i3 = ((idx /= 3    ) % 3);   //s
    const long i4 = ((idx /= 3    ) % 512);   //c
    const long i5 = ((idx /= 512    ) % 512);   //k
    const long i6 = ((idx /= 512    ) % 14);   //p
    const long i7 = ((idx /= 14    ) % 14);   //q

    const long h = i6 * 1 - 1 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 1 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l83(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]) { 
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

  const long N1 = 64L*512*14*14;  // N * C * H * W
  const long N2 = 512L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

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
  const long N_bn = 64L*14*14;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l82(float input[64][1024][14][14], float d_output[64][512][14][14], float d_weight[512][1024][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 512L*1024*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);   // k
    const long i2 = ((idx /= 512) % 1024);   // c
    const long i3 = ((idx /= 1024) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*1024*512*14*14;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 1024); //c
    const long i5 = ((idx /= 1024  ) % 512); //k
    const long i6 = ((idx /= 512  ) % 14); //p
    const long i7 = ((idx /= 14  ) % 14); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l82(float d_input[64][1024][14][14], float d_output[64][512][14][14], float weight[512][1024][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*1024*14*14;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 1024);  //c
    const long i3 = ((idx /= 1024   ) % 14);  //h
    const long i4 = ((idx /= 14   ) % 14);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*1024*512*14*14;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 1024);   //c
    const long i5 = ((idx /= 1024    ) % 512);   //k
    const long i6 = ((idx /= 512    ) % 14);   //p
    const long i7 = ((idx /= 14    ) % 14);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void multiGradient_Add_l81(float* d_output0, float* d_output1, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        d_output0[idx] += d_output0[idx] + d_output1[idx];
      }
  }
}


__global__ void backwardPass_ReLU_l80(float d_input[64][1024][14][14], float input[64][1024][14][14], float d_output[64][1024][14][14]) {

  long size = 64L*1024*14*14;  //N * C * H * W;
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;
  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);  //n
    const long i2 = ((idx /= 64    ) % 1024);  //c
    const long i3 = ((idx /= 1024    ) % 14);  //h
    const long i4 = ((idx /= 14    ) % 14);  //w

    d_input[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? d_output[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void backwardPass_BatchNorm2d_l79(float d_output[64][1024][14][14], float v1[64][1024][14][14], float v2[1024], float gamma_and_beta[2048], float dv1[64][1024][14][14], float dv2[1024], float dvar[1024], float dmu[1024], float d_input[64][1024][14][14], float d_gamma_and_beta[2048]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 1024L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[1024 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*1024*14*14;  // N * C * H * W
  const long N2 = 1024L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*14*14;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l78(float input[64][512][14][14], float d_output[64][1024][14][14], float d_weight[1024][512][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 1024L*512*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);   // k
    const long i2 = ((idx /= 1024) % 512);   // c
    const long i3 = ((idx /= 512) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*512*1024*14*14;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 512); //c
    const long i5 = ((idx /= 512  ) % 1024); //k
    const long i6 = ((idx /= 1024  ) % 14); //p
    const long i7 = ((idx /= 14  ) % 14); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l78(float d_input[64][512][14][14], float d_output[64][1024][14][14], float weight[1024][512][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*512*14*14;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 512);  //c
    const long i3 = ((idx /= 512   ) % 14);  //h
    const long i4 = ((idx /= 14   ) % 14);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*512*1024*14*14;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 512);   //c
    const long i5 = ((idx /= 512    ) % 1024);   //k
    const long i6 = ((idx /= 1024    ) % 14);   //p
    const long i7 = ((idx /= 14    ) % 14);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l77(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]) { 
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

  const long N1 = 64L*512*14*14;  // N * C * H * W
  const long N2 = 512L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

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
  const long N_bn = 64L*14*14;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l76(float input[64][512][14][14], float d_output[64][512][14][14], float d_weight[512][512][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 512L*512*3*3;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);   // k
    const long i2 = ((idx /= 512) % 512);   // c
    const long i3 = ((idx /= 512) % 3);   // r
    const long i4 = ((idx /= 3) % 3);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*3*3*512*512*14*14;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 3); //r
    const long i3 = ((idx /= 3  ) % 3); //s
    const long i4 = ((idx /= 3  ) % 512); //c
    const long i5 = ((idx /= 512  ) % 512); //k
    const long i6 = ((idx /= 512  ) % 14); //p
    const long i7 = ((idx /= 14  ) % 14); //q

    const long h = i6 * 1 - 1 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 1 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l76(float d_input[64][512][14][14], float d_output[64][512][14][14], float weight[512][512][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*512*14*14;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 512);  //c
    const long i3 = ((idx /= 512   ) % 14);  //h
    const long i4 = ((idx /= 14   ) % 14);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*3*3*512*512*14*14;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 3);   //r
    const long i3 = ((idx /= 3    ) % 3);   //s
    const long i4 = ((idx /= 3    ) % 512);   //c
    const long i5 = ((idx /= 512    ) % 512);   //k
    const long i6 = ((idx /= 512    ) % 14);   //p
    const long i7 = ((idx /= 14    ) % 14);   //q

    const long h = i6 * 1 - 1 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 1 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l75(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]) { 
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

  const long N1 = 64L*512*14*14;  // N * C * H * W
  const long N2 = 512L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

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
  const long N_bn = 64L*14*14;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l74(float input[64][1024][14][14], float d_output[64][512][14][14], float d_weight[512][1024][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 512L*1024*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);   // k
    const long i2 = ((idx /= 512) % 1024);   // c
    const long i3 = ((idx /= 1024) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*1024*512*14*14;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 1024); //c
    const long i5 = ((idx /= 1024  ) % 512); //k
    const long i6 = ((idx /= 512  ) % 14); //p
    const long i7 = ((idx /= 14  ) % 14); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l74(float d_input[64][1024][14][14], float d_output[64][512][14][14], float weight[512][1024][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*1024*14*14;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 1024);  //c
    const long i3 = ((idx /= 1024   ) % 14);  //h
    const long i4 = ((idx /= 14   ) % 14);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*1024*512*14*14;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 1024);   //c
    const long i5 = ((idx /= 1024    ) % 512);   //k
    const long i6 = ((idx /= 512    ) % 14);   //p
    const long i7 = ((idx /= 14    ) % 14);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void multiGradient_Add_l73(float* d_output0, float* d_output1, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        d_output0[idx] += d_output0[idx] + d_output1[idx];
      }
  }
}


__global__ void backwardPass_BatchNorm2d_l72(float d_output[64][1024][14][14], float v1[64][1024][14][14], float v2[1024], float gamma_and_beta[2048], float dv1[64][1024][14][14], float dv2[1024], float dvar[1024], float dmu[1024], float d_input[64][1024][14][14], float d_gamma_and_beta[2048]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 1024L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[1024 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*1024*14*14;  // N * C * H * W
  const long N2 = 1024L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*14*14;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l71(float input[64][512][28][28], float d_output[64][1024][14][14], float d_weight[1024][512][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 1024L*512*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);   // k
    const long i2 = ((idx /= 1024) % 512);   // c
    const long i3 = ((idx /= 512) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*512*1024*14*14;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 512); //c
    const long i5 = ((idx /= 512  ) % 1024); //k
    const long i6 = ((idx /= 1024  ) % 14); //p
    const long i7 = ((idx /= 14  ) % 14); //q

    const long h = i6 * 2 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 2 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l71(float d_input[64][512][28][28], float d_output[64][1024][14][14], float weight[1024][512][1][1])
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

  const long N = 64L*1*1*512*1024*14*14;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 512);   //c
    const long i5 = ((idx /= 512    ) % 1024);   //k
    const long i6 = ((idx /= 1024    ) % 14);   //p
    const long i7 = ((idx /= 14    ) % 14);   //q

    const long h = i6 * 2 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 2 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_ReLU_l70(float d_input[64][1024][14][14], float input[64][1024][14][14], float d_output[64][1024][14][14]) {

  long size = 64L*1024*14*14;  //N * C * H * W;
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;
  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);  //n
    const long i2 = ((idx /= 64    ) % 1024);  //c
    const long i3 = ((idx /= 1024    ) % 14);  //h
    const long i4 = ((idx /= 14    ) % 14);  //w

    d_input[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? d_output[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void backwardPass_BatchNorm2d_l69(float d_output[64][1024][14][14], float v1[64][1024][14][14], float v2[1024], float gamma_and_beta[2048], float dv1[64][1024][14][14], float dv2[1024], float dvar[1024], float dmu[1024], float d_input[64][1024][14][14], float d_gamma_and_beta[2048]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 1024L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[1024 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*1024*14*14;  // N * C * H * W
  const long N2 = 1024L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*14*14;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l68(float input[64][512][14][14], float d_output[64][1024][14][14], float d_weight[1024][512][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 1024L*512*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);   // k
    const long i2 = ((idx /= 1024) % 512);   // c
    const long i3 = ((idx /= 512) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*512*1024*14*14;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 512); //c
    const long i5 = ((idx /= 512  ) % 1024); //k
    const long i6 = ((idx /= 1024  ) % 14); //p
    const long i7 = ((idx /= 14  ) % 14); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l68(float d_input[64][512][14][14], float d_output[64][1024][14][14], float weight[1024][512][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*512*14*14;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 512);  //c
    const long i3 = ((idx /= 512   ) % 14);  //h
    const long i4 = ((idx /= 14   ) % 14);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*512*1024*14*14;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 512);   //c
    const long i5 = ((idx /= 512    ) % 1024);   //k
    const long i6 = ((idx /= 1024    ) % 14);   //p
    const long i7 = ((idx /= 14    ) % 14);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 14 && w >= 0 && w < 14)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l67(float d_output[64][512][14][14], float v1[64][512][14][14], float v2[512], float gamma_and_beta[1024], float dv1[64][512][14][14], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][14][14], float d_gamma_and_beta[1024]) { 
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

  const long N1 = 64L*512*14*14;  // N * C * H * W
  const long N2 = 512L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

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
  const long N_bn = 64L*14*14;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l66(float input[64][512][28][28], float d_output[64][512][14][14], float d_weight[512][512][3][3])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 512L*512*3*3;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);   // k
    const long i2 = ((idx /= 512) % 512);   // c
    const long i3 = ((idx /= 512) % 3);   // r
    const long i4 = ((idx /= 3) % 3);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*3*3*512*512*14*14;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 3); //r
    const long i3 = ((idx /= 3  ) % 3); //s
    const long i4 = ((idx /= 3  ) % 512); //c
    const long i5 = ((idx /= 512  ) % 512); //k
    const long i6 = ((idx /= 512  ) % 14); //p
    const long i7 = ((idx /= 14  ) % 14); //q

    const long h = i6 * 2 - 1 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 2 - 1 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l66(float d_input[64][512][28][28], float d_output[64][512][14][14], float weight[512][512][3][3])
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

  const long N = 64L*3*3*512*512*14*14;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 3);   //r
    const long i3 = ((idx /= 3    ) % 3);   //s
    const long i4 = ((idx /= 3    ) % 512);   //c
    const long i5 = ((idx /= 512    ) % 512);   //k
    const long i6 = ((idx /= 512    ) % 14);   //p
    const long i7 = ((idx /= 14    ) % 14);   //q

    const long h = i6 * 2 - 1 + i2; // h = p * stride - pad + r;
    const long w = i7 * 2 - 1 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 28 && w >= 0 && w < 28)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

__global__ void backwardPass_BatchNorm2d_l65(float d_output[64][512][28][28], float v1[64][512][28][28], float v2[512], float gamma_and_beta[1024], float dv1[64][512][28][28], float dv2[512], float dvar[512], float dmu[512], float d_input[64][512][28][28], float d_gamma_and_beta[1024]) { 
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

__global__ void backwardPass_conv2d_Weight_l64(float input[64][512][28][28], float d_output[64][512][28][28], float d_weight[512][512][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 512L*512*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);   // k
    const long i2 = ((idx /= 512) % 512);   // c
    const long i3 = ((idx /= 512) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*512*512*28*28;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 512); //c
    const long i5 = ((idx /= 512  ) % 512); //k
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

__global__ void backwardPass_conv2d_Input_l64(float d_input[64][512][28][28], float d_output[64][512][28][28], float weight[512][512][1][1])
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

  const long N = 64L*1*1*512*512*28*28;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 512);   //c
    const long i5 = ((idx /= 512    ) % 512);   //k
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

