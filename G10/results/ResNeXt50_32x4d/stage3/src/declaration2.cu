__global__ void forwardPass_conv2d_l100(float input[64][512][14][14], float output[64][512][14][14], float weight[512][512][3][3])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*512*14*14;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 512);     // k
    const long i3 = ((idx /= 512) % 14);     // p
    const long i4 = ((idx /= 14) % 14);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*3*3*512*512*14*14;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 3);     // r
		const long i3 = ((idx /= 3) % 3);     // s
		const long i4 = ((idx /= 3) % 512);     // c
		const long i5 = ((idx /= 512) % 512);     // k
		const long i6 = ((idx /= 512) % 14);     // p
		const long i7 = ((idx /= 14) % 14);     // q
		const long h = i6 * 1 - 1 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 1 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 14 && w >= 0 && w < 14)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l101(float input[64][512][14][14], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][14][14], float v2[512], float output[64][512][14][14]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 512L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*512*14*14;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*14*14;  // N * H * W
  const long N2 = 512; // C
  
  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      mu[idx] = mu[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_m[idx] = 0.100000 * running_m[idx] + 0.900000 * mu[idx];    // running_m = momentum * running_m + (1-momentum) * mu;
    }
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&var[i2], (v1[i1][i2][i3][i4]*v1[i1][i2][i3][i4]));

  }

  __syncthreads();


  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      var[idx] = var[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_v[idx] = 0.100000 * running_v[idx] + 0.900000 * var[idx];   // running_v = momentum * running_v + (1-momentum) * var;
    }
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[512 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l102(float input[64][512][14][14], float output[64][1024][14][14], float weight[1024][512][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*1024*14*14;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 1024);     // k
    const long i3 = ((idx /= 1024) % 14);     // p
    const long i4 = ((idx /= 14) % 14);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*512*1024*14*14;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 512);     // c
		const long i5 = ((idx /= 512) % 1024);     // k
		const long i6 = ((idx /= 1024) % 14);     // p
		const long i7 = ((idx /= 14) % 14);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 14 && w >= 0 && w < 14)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l103(float input[64][1024][14][14], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][14][14], float v2[1024], float output[64][1024][14][14]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 1024L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*1024*14*14;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*14*14;  // N * H * W
  const long N2 = 1024; // C
  
  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      mu[idx] = mu[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_m[idx] = 0.100000 * running_m[idx] + 0.900000 * mu[idx];    // running_m = momentum * running_m + (1-momentum) * mu;
    }
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&var[i2], (v1[i1][i2][i3][i4]*v1[i1][i2][i3][i4]));

  }

  __syncthreads();


  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      var[idx] = var[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_v[idx] = 0.100000 * running_v[idx] + 0.900000 * var[idx];   // running_v = momentum * running_v + (1-momentum) * var;
    }
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[1024 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_ReLU_l104(float input[64][1024][14][14], float output[64][1024][14][14]){

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

    output[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? input[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void forwardPass_Add_l105(float* input0, float* input1, float* out, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        out[idx] = input0[idx] + input1[idx];
      }
  }
}


__global__ void forwardPass_conv2d_l106(float input[64][1024][14][14], float output[64][512][14][14], float weight[512][1024][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*512*14*14;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 512);     // k
    const long i3 = ((idx /= 512) % 14);     // p
    const long i4 = ((idx /= 14) % 14);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*1024*512*14*14;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 1024);     // c
		const long i5 = ((idx /= 1024) % 512);     // k
		const long i6 = ((idx /= 512) % 14);     // p
		const long i7 = ((idx /= 14) % 14);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 14 && w >= 0 && w < 14)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l107(float input[64][512][14][14], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][14][14], float v2[512], float output[64][512][14][14]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 512L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*512*14*14;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*14*14;  // N * H * W
  const long N2 = 512; // C
  
  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      mu[idx] = mu[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_m[idx] = 0.100000 * running_m[idx] + 0.900000 * mu[idx];    // running_m = momentum * running_m + (1-momentum) * mu;
    }
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&var[i2], (v1[i1][i2][i3][i4]*v1[i1][i2][i3][i4]));

  }

  __syncthreads();


  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      var[idx] = var[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_v[idx] = 0.100000 * running_v[idx] + 0.900000 * var[idx];   // running_v = momentum * running_v + (1-momentum) * var;
    }
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[512 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l108(float input[64][512][14][14], float output[64][512][14][14], float weight[512][512][3][3])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*512*14*14;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 512);     // k
    const long i3 = ((idx /= 512) % 14);     // p
    const long i4 = ((idx /= 14) % 14);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*3*3*512*512*14*14;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 3);     // r
		const long i3 = ((idx /= 3) % 3);     // s
		const long i4 = ((idx /= 3) % 512);     // c
		const long i5 = ((idx /= 512) % 512);     // k
		const long i6 = ((idx /= 512) % 14);     // p
		const long i7 = ((idx /= 14) % 14);     // q
		const long h = i6 * 1 - 1 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 1 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 14 && w >= 0 && w < 14)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l109(float input[64][512][14][14], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][14][14], float v2[512], float output[64][512][14][14]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 512L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*512*14*14;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*14*14;  // N * H * W
  const long N2 = 512; // C
  
  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      mu[idx] = mu[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_m[idx] = 0.100000 * running_m[idx] + 0.900000 * mu[idx];    // running_m = momentum * running_m + (1-momentum) * mu;
    }
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&var[i2], (v1[i1][i2][i3][i4]*v1[i1][i2][i3][i4]));

  }

  __syncthreads();


  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      var[idx] = var[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_v[idx] = 0.100000 * running_v[idx] + 0.900000 * var[idx];   // running_v = momentum * running_v + (1-momentum) * var;
    }
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 512);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[512 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l110(float input[64][512][14][14], float output[64][1024][14][14], float weight[1024][512][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*1024*14*14;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 1024);     // k
    const long i3 = ((idx /= 1024) % 14);     // p
    const long i4 = ((idx /= 14) % 14);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*512*1024*14*14;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 512);     // c
		const long i5 = ((idx /= 512) % 1024);     // k
		const long i6 = ((idx /= 1024) % 14);     // p
		const long i7 = ((idx /= 14) % 14);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 14 && w >= 0 && w < 14)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l111(float input[64][1024][14][14], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][14][14], float v2[1024], float output[64][1024][14][14]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 1024L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*1024*14*14;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*14*14;  // N * H * W
  const long N2 = 1024; // C
  
  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      mu[idx] = mu[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_m[idx] = 0.100000 * running_m[idx] + 0.900000 * mu[idx];    // running_m = momentum * running_m + (1-momentum) * mu;
    }
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&var[i2], (v1[i1][i2][i3][i4]*v1[i1][i2][i3][i4]));

  }

  __syncthreads();


  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      var[idx] = var[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_v[idx] = 0.100000 * running_v[idx] + 0.900000 * var[idx];   // running_v = momentum * running_v + (1-momentum) * var;
    }
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[1024 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_ReLU_l112(float input[64][1024][14][14], float output[64][1024][14][14]){

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

    output[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? input[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void forwardPass_Add_l113(float* input0, float* input1, float* out, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        out[idx] = input0[idx] + input1[idx];
      }
  }
}


__global__ void forwardPass_conv2d_l114(float input[64][1024][14][14], float output[64][1024][14][14], float weight[1024][1024][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*1024*14*14;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 1024);     // k
    const long i3 = ((idx /= 1024) % 14);     // p
    const long i4 = ((idx /= 14) % 14);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*1024*1024*14*14;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 1024);     // c
		const long i5 = ((idx /= 1024) % 1024);     // k
		const long i6 = ((idx /= 1024) % 14);     // p
		const long i7 = ((idx /= 14) % 14);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 14 && w >= 0 && w < 14)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l115(float input[64][1024][14][14], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][14][14], float v2[1024], float output[64][1024][14][14]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 1024L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*1024*14*14;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*14*14;  // N * H * W
  const long N2 = 1024; // C
  
  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      mu[idx] = mu[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_m[idx] = 0.100000 * running_m[idx] + 0.900000 * mu[idx];    // running_m = momentum * running_m + (1-momentum) * mu;
    }
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    atomicAdd(&var[i2], (v1[i1][i2][i3][i4]*v1[i1][i2][i3][i4]));

  }

  __syncthreads();


  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      var[idx] = var[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_v[idx] = 0.100000 * running_v[idx] + 0.900000 * var[idx];   // running_v = momentum * running_v + (1-momentum) * var;
    }
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 14); // h
    const long i4 = ((idx /= 14) % 14); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[1024 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l116(float input[64][1024][14][14], float output[64][1024][7][7], float weight[1024][1024][3][3])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*1024*7*7;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 1024);     // k
    const long i3 = ((idx /= 1024) % 7);     // p
    const long i4 = ((idx /= 7) % 7);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*3*3*1024*1024*7*7;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 3);     // r
		const long i3 = ((idx /= 3) % 3);     // s
		const long i4 = ((idx /= 3) % 1024);     // c
		const long i5 = ((idx /= 1024) % 1024);     // k
		const long i6 = ((idx /= 1024) % 7);     // p
		const long i7 = ((idx /= 7) % 7);     // q
		const long h = i6 * 2 - 1 + i2; // h = p * stride - pad + r;
		const long w = i7 * 2 - 1 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 14 && w >= 0 && w < 14)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l117(float input[64][1024][7][7], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][7][7], float v2[1024], float output[64][1024][7][7]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 1024L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*1024*7*7;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*7*7;  // N * H * W
  const long N2 = 1024; // C
  
  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      mu[idx] = mu[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_m[idx] = 0.100000 * running_m[idx] + 0.900000 * mu[idx];    // running_m = momentum * running_m + (1-momentum) * mu;
    }
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&var[i2], (v1[i1][i2][i3][i4]*v1[i1][i2][i3][i4]));

  }

  __syncthreads();


  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      var[idx] = var[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_v[idx] = 0.100000 * running_v[idx] + 0.900000 * var[idx];   // running_v = momentum * running_v + (1-momentum) * var;
    }
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[1024 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l118(float input[64][1024][7][7], float output[64][2048][7][7], float weight[2048][1024][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*2048*7*7;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 2048);     // k
    const long i3 = ((idx /= 2048) % 7);     // p
    const long i4 = ((idx /= 7) % 7);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*1024*2048*7*7;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 1024);     // c
		const long i5 = ((idx /= 1024) % 2048);     // k
		const long i6 = ((idx /= 2048) % 7);     // p
		const long i7 = ((idx /= 7) % 7);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 7 && w >= 0 && w < 7)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l119(float input[64][2048][7][7], float gamma_and_beta[4096], float running_m[2048], float running_v[2048], float mu[2048], float var[2048], float v1[64][2048][7][7], float v2[2048], float output[64][2048][7][7]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 2048L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 2048);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*2048*7*7;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*7*7;  // N * H * W
  const long N2 = 2048; // C
  
  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      mu[idx] = mu[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_m[idx] = 0.100000 * running_m[idx] + 0.900000 * mu[idx];    // running_m = momentum * running_m + (1-momentum) * mu;
    }
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&var[i2], (v1[i1][i2][i3][i4]*v1[i1][i2][i3][i4]));

  }

  __syncthreads();


  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      var[idx] = var[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_v[idx] = 0.100000 * running_v[idx] + 0.900000 * var[idx];   // running_v = momentum * running_v + (1-momentum) * var;
    }
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 2048);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[2048 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_ReLU_l120(float input[64][2048][7][7], float output[64][2048][7][7]){

  long size = 64L*2048*7*7;  //N * C * H * W;
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);  //n
    const long i2 = ((idx /= 64    ) % 2048);  //c
    const long i3 = ((idx /= 2048    ) % 7);  //h
    const long i4 = ((idx /= 7    ) % 7);  //w

    output[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? input[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void forwardPass_conv2d_l121(float input[64][1024][14][14], float output[64][2048][7][7], float weight[2048][1024][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*2048*7*7;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 2048);     // k
    const long i3 = ((idx /= 2048) % 7);     // p
    const long i4 = ((idx /= 7) % 7);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*1024*2048*7*7;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 1024);     // c
		const long i5 = ((idx /= 1024) % 2048);     // k
		const long i6 = ((idx /= 2048) % 7);     // p
		const long i7 = ((idx /= 7) % 7);     // q
		const long h = i6 * 2 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 2 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 14 && w >= 0 && w < 14)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l122(float input[64][2048][7][7], float gamma_and_beta[4096], float running_m[2048], float running_v[2048], float mu[2048], float var[2048], float v1[64][2048][7][7], float v2[2048], float output[64][2048][7][7]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 2048L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 2048);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*2048*7*7;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*7*7;  // N * H * W
  const long N2 = 2048; // C
  
  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      mu[idx] = mu[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_m[idx] = 0.100000 * running_m[idx] + 0.900000 * mu[idx];    // running_m = momentum * running_m + (1-momentum) * mu;
    }
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&var[i2], (v1[i1][i2][i3][i4]*v1[i1][i2][i3][i4]));

  }

  __syncthreads();


  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      var[idx] = var[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_v[idx] = 0.100000 * running_v[idx] + 0.900000 * var[idx];   // running_v = momentum * running_v + (1-momentum) * var;
    }
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 2048);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[2048 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_Add_l123(float* input0, float* input1, float* out, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        out[idx] = input0[idx] + input1[idx];
      }
  }
}


__global__ void forwardPass_conv2d_l124(float input[64][2048][7][7], float output[64][1024][7][7], float weight[1024][2048][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*1024*7*7;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 1024);     // k
    const long i3 = ((idx /= 1024) % 7);     // p
    const long i4 = ((idx /= 7) % 7);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*2048*1024*7*7;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 2048);     // c
		const long i5 = ((idx /= 2048) % 1024);     // k
		const long i6 = ((idx /= 1024) % 7);     // p
		const long i7 = ((idx /= 7) % 7);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 7 && w >= 0 && w < 7)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l125(float input[64][1024][7][7], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][7][7], float v2[1024], float output[64][1024][7][7]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 1024L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*1024*7*7;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*7*7;  // N * H * W
  const long N2 = 1024; // C
  
  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      mu[idx] = mu[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_m[idx] = 0.100000 * running_m[idx] + 0.900000 * mu[idx];    // running_m = momentum * running_m + (1-momentum) * mu;
    }
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&var[i2], (v1[i1][i2][i3][i4]*v1[i1][i2][i3][i4]));

  }

  __syncthreads();


  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      var[idx] = var[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_v[idx] = 0.100000 * running_v[idx] + 0.900000 * var[idx];   // running_v = momentum * running_v + (1-momentum) * var;
    }
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[1024 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l126(float input[64][1024][7][7], float output[64][1024][7][7], float weight[1024][1024][3][3])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*1024*7*7;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 1024);     // k
    const long i3 = ((idx /= 1024) % 7);     // p
    const long i4 = ((idx /= 7) % 7);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*3*3*1024*1024*7*7;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 3);     // r
		const long i3 = ((idx /= 3) % 3);     // s
		const long i4 = ((idx /= 3) % 1024);     // c
		const long i5 = ((idx /= 1024) % 1024);     // k
		const long i6 = ((idx /= 1024) % 7);     // p
		const long i7 = ((idx /= 7) % 7);     // q
		const long h = i6 * 1 - 1 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 1 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 7 && w >= 0 && w < 7)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l127(float input[64][1024][7][7], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][7][7], float v2[1024], float output[64][1024][7][7]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 1024L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*1024*7*7;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*7*7;  // N * H * W
  const long N2 = 1024; // C
  
  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      mu[idx] = mu[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_m[idx] = 0.100000 * running_m[idx] + 0.900000 * mu[idx];    // running_m = momentum * running_m + (1-momentum) * mu;
    }
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&var[i2], (v1[i1][i2][i3][i4]*v1[i1][i2][i3][i4]));

  }

  __syncthreads();


  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      var[idx] = var[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_v[idx] = 0.100000 * running_v[idx] + 0.900000 * var[idx];   // running_v = momentum * running_v + (1-momentum) * var;
    }
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[1024 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l128(float input[64][1024][7][7], float output[64][2048][7][7], float weight[2048][1024][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*2048*7*7;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 2048);     // k
    const long i3 = ((idx /= 2048) % 7);     // p
    const long i4 = ((idx /= 7) % 7);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*1024*2048*7*7;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 1024);     // c
		const long i5 = ((idx /= 1024) % 2048);     // k
		const long i6 = ((idx /= 2048) % 7);     // p
		const long i7 = ((idx /= 7) % 7);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 7 && w >= 0 && w < 7)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l129(float input[64][2048][7][7], float gamma_and_beta[4096], float running_m[2048], float running_v[2048], float mu[2048], float var[2048], float v1[64][2048][7][7], float v2[2048], float output[64][2048][7][7]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 2048L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 2048);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*2048*7*7;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*7*7;  // N * H * W
  const long N2 = 2048; // C
  
  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      mu[idx] = mu[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_m[idx] = 0.100000 * running_m[idx] + 0.900000 * mu[idx];    // running_m = momentum * running_m + (1-momentum) * mu;
    }
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&var[i2], (v1[i1][i2][i3][i4]*v1[i1][i2][i3][i4]));

  }

  __syncthreads();


  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      var[idx] = var[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_v[idx] = 0.100000 * running_v[idx] + 0.900000 * var[idx];   // running_v = momentum * running_v + (1-momentum) * var;
    }
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 2048);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[2048 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_ReLU_l130(float input[64][2048][7][7], float output[64][2048][7][7]){

  long size = 64L*2048*7*7;  //N * C * H * W;
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);  //n
    const long i2 = ((idx /= 64    ) % 2048);  //c
    const long i3 = ((idx /= 2048    ) % 7);  //h
    const long i4 = ((idx /= 7    ) % 7);  //w

    output[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? input[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void forwardPass_Add_l131(float* input0, float* input1, float* out, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        out[idx] = input0[idx] + input1[idx];
      }
  }
}


__global__ void forwardPass_conv2d_l132(float input[64][2048][7][7], float output[64][1024][7][7], float weight[1024][2048][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*1024*7*7;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 1024);     // k
    const long i3 = ((idx /= 1024) % 7);     // p
    const long i4 = ((idx /= 7) % 7);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*2048*1024*7*7;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 2048);     // c
		const long i5 = ((idx /= 2048) % 1024);     // k
		const long i6 = ((idx /= 1024) % 7);     // p
		const long i7 = ((idx /= 7) % 7);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 7 && w >= 0 && w < 7)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l133(float input[64][1024][7][7], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][7][7], float v2[1024], float output[64][1024][7][7]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 1024L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*1024*7*7;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*7*7;  // N * H * W
  const long N2 = 1024; // C
  
  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      mu[idx] = mu[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_m[idx] = 0.100000 * running_m[idx] + 0.900000 * mu[idx];    // running_m = momentum * running_m + (1-momentum) * mu;
    }
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&var[i2], (v1[i1][i2][i3][i4]*v1[i1][i2][i3][i4]));

  }

  __syncthreads();


  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      var[idx] = var[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_v[idx] = 0.100000 * running_v[idx] + 0.900000 * var[idx];   // running_v = momentum * running_v + (1-momentum) * var;
    }
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[1024 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l134(float input[64][1024][7][7], float output[64][1024][7][7], float weight[1024][1024][3][3])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*1024*7*7;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 1024);     // k
    const long i3 = ((idx /= 1024) % 7);     // p
    const long i4 = ((idx /= 7) % 7);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*3*3*1024*1024*7*7;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 3);     // r
		const long i3 = ((idx /= 3) % 3);     // s
		const long i4 = ((idx /= 3) % 1024);     // c
		const long i5 = ((idx /= 1024) % 1024);     // k
		const long i6 = ((idx /= 1024) % 7);     // p
		const long i7 = ((idx /= 7) % 7);     // q
		const long h = i6 * 1 - 1 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 1 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 7 && w >= 0 && w < 7)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l135(float input[64][1024][7][7], float gamma_and_beta[2048], float running_m[1024], float running_v[1024], float mu[1024], float var[1024], float v1[64][1024][7][7], float v2[1024], float output[64][1024][7][7]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 1024L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*1024*7*7;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*7*7;  // N * H * W
  const long N2 = 1024; // C
  
  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      mu[idx] = mu[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_m[idx] = 0.100000 * running_m[idx] + 0.900000 * mu[idx];    // running_m = momentum * running_m + (1-momentum) * mu;
    }
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&var[i2], (v1[i1][i2][i3][i4]*v1[i1][i2][i3][i4]));

  }

  __syncthreads();


  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      var[idx] = var[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_v[idx] = 0.100000 * running_v[idx] + 0.900000 * var[idx];   // running_v = momentum * running_v + (1-momentum) * var;
    }
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 1024);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 1024); // c
    const long i3 = ((idx /= 1024) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[1024 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l136(float input[64][1024][7][7], float output[64][2048][7][7], float weight[2048][1024][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*2048*7*7;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 2048);     // k
    const long i3 = ((idx /= 2048) % 7);     // p
    const long i4 = ((idx /= 7) % 7);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*1024*2048*7*7;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 1024);     // c
		const long i5 = ((idx /= 1024) % 2048);     // k
		const long i6 = ((idx /= 2048) % 7);     // p
		const long i7 = ((idx /= 7) % 7);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 7 && w >= 0 && w < 7)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l137(float input[64][2048][7][7], float gamma_and_beta[4096], float running_m[2048], float running_v[2048], float mu[2048], float var[2048], float v1[64][2048][7][7], float v2[2048], float output[64][2048][7][7]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 2048L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 2048);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*2048*7*7;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*7*7;  // N * H * W
  const long N2 = 2048; // C
  
  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      mu[idx] = mu[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_m[idx] = 0.100000 * running_m[idx] + 0.900000 * mu[idx];    // running_m = momentum * running_m + (1-momentum) * mu;
    }
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&var[i2], (v1[i1][i2][i3][i4]*v1[i1][i2][i3][i4]));

  }

  __syncthreads();


  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      var[idx] = var[idx] / N_bn;
    }
  }

  __syncthreads();

  for (long n = 0; n < N2; n += parallel_size) {
    long idx = n + thread_pos;
    if (idx < N2) {
      running_v[idx] = 0.100000 * running_v[idx] + 0.900000 * var[idx];   // running_v = momentum * running_v + (1-momentum) * var;
    }
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 2048);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[2048 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_ReLU_l138(float input[64][2048][7][7], float output[64][2048][7][7]){

  long size = 64L*2048*7*7;  //N * C * H * W;
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);  //n
    const long i2 = ((idx /= 64    ) % 2048);  //c
    const long i3 = ((idx /= 2048    ) % 7);  //h
    const long i4 = ((idx /= 7    ) % 7);  //w

    output[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? input[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void forwardPass_Add_l139(float* input0, float* input1, float* out, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        out[idx] = input0[idx] + input1[idx];
      }
  }
}


__global__ void forwardPass_avgAdaptivePool2d_l140(float input[64][2048][7][7], float output[64][2048][1][1]) {

  long size = 64L*2048*1*1;  //N * C * S0 * S1;
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;
  const long ph = 7;   // H/S0
  const long pw = 7;   // W/S1

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);  //n
    const long i2 = ((idx /= 64    ) % 2048);  //c
    const long i3 = ((idx /= 2048    ) % 1);  //p
    const long i4 = ((idx /= 1    ) % 1);  //q

    float sum = 0;

    for (long i = 0; i < ph; i++)
    {
      for (long j = 0; j < pw; j++)
      {
        sum += input[i1][i2][i3*ph+i][i4*pw+j];
      }
    }

    output[i1][i2][i3][i4] = sum / (ph*pw);
  }
}

__global__ void forwardPass_Linear_l141(float input[64][2048], float output[64][1000], float weight[2048][1000], float bias[1000]) {
  long size = 64L*1000;  //N*H_out
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1)  % 64);  //n
    const long i2 = ((idx /= 64) % 1000);  //h_out

    output[i1][i2] = 0;
    for (long k = 0; k < 2048; ++k) {
      output[i1][i2] += input[i1][k] * weight[k][i2];
    }

    output[i1][i2] += bias[i2];
  }
}

__global__ void backwardPass_Linear_bias_l141(float d_output[64][1000], float d_bias[1000]) {
  long size = 1000L;  //H_out
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    d_bias[n] = 0;

    for (long i = 0; i < 64; ++i) {
      d_bias[n] += d_output[i][n];
    }
  }
}

__global__ void backwardPass_Linear_weight_l141(float input[64][2048], float d_output[64][1000], float d_weight[2048][1000]) {

  long size = 2048L*1000;  //H_in * H_out
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 2048);  //H_in
    const long i2 = ((idx /= 2048   ) % 1000);  //H_out
    d_weight[i1][i2] = 0;

    for (long k = 0; k < 64; ++k) {
      d_weight[i1][i2] += input[k][i1] * d_output[k][i2];
    }
  }
}

__global__ void backwardPass_Linear_input_l141(float d_input[64][2048], float d_output[64][1000], float weight[2048][1000]) {
  long size = 64L*2048;  //N * H_in
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);  //n
    const long i2 = ((idx /= 64) % 2048); //h_in

    d_input[i1][i2] = 0;
    for (long k = 0; k < 1000; ++k) {
      d_input[i1][i2] += d_output[i1][k] * weight[i2][k];
    }
  }
}

__global__ void backwardPass_avgAdaptivePool2d_l140(float d_input[64][2048][7][7], float d_output[64][2048][1][1]) {
  long size = 64L*2048*7*7;  //N * C * H * W
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;
  const long ph = 7;   // H/S0
  const long pw = 7;   // W/S1

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);  //n
    const long i2 = ((idx /= 64    ) % 2048);  //c
    const long i3 = ((idx /= 2048    ) % 7);  //h
    const long i4 = ((idx /= 7    ) % 7);  //w

    d_input[i1][i2][i3][i4] = d_output[i1][i2][i3/ph][i4/ph] / (ph*pw);
  }
}

__global__ void backwardPass_ReLU_l138(float d_input[64][2048][7][7], float input[64][2048][7][7], float d_output[64][2048][7][7]) {

  long size = 64L*2048*7*7;  //N * C * H * W;
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = size;
  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);  //n
    const long i2 = ((idx /= 64    ) % 2048);  //c
    const long i3 = ((idx /= 2048    ) % 7);  //h
    const long i4 = ((idx /= 7    ) % 7);  //w

    d_input[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? d_output[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void backwardPass_BatchNorm2d_l137(float d_output[64][2048][7][7], float v1[64][2048][7][7], float v2[2048], float gamma_and_beta[4096], float dv1[64][2048][7][7], float dv2[2048], float dvar[2048], float dmu[2048], float d_input[64][2048][7][7], float d_gamma_and_beta[4096]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 2048L;  // C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 2048);  // c

    d_gamma_and_beta[i1] = 0;
    d_gamma_and_beta[2048 + i1] = 0;     // C + i1
  }

  __syncthreads();

  const long N1 = 64L*2048*7*7;  // N * C * H * W
  const long N2 = 2048L; // C

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&d_gamma_and_beta[i2], d_output[i1][i2][i3][i4]);
    atomicAdd(&d_gamma_and_beta[6+i2], (v1[i1][i2][i3][i4]/v2[i2]) * d_output[i1][i2][i3][i4]);
    
  }
  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    float dx_bar = gamma_and_beta[i2] * d_output[i1][i2][i3][i4];

    dv1[i1][i2][i3][i4] = (dx_bar) / v2[i2];
    atomicAdd(&dv2[i2], (-1) * dx_bar / (v2[i2]*v2[i2]));
  }

  __syncthreads();

  for (long n = N2 * thread_pos / parallel_size; n < N2 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 2048);  // c

    dvar[i1] = dv2[i1] / (2 * v2[i1]);
  }

  __syncthreads();
  const long N_bn = 64L*7*7;  // N * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&dv1[i1][i2][i3][i4], 2 * v1[i1][i2][i3][i4] * dvar[i2] / N_bn);
  }

  __syncthreads();

  

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n){
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    atomicAdd(&dmu[i2], -dv1[i1][i2][i3][i4]);
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 2048); // c
    const long i3 = ((idx /= 2048) % 7); // h
    const long i4 = ((idx /= 7) % 7); // w

    d_input[i1][i2][i3][i4] = dv1[i1][i2][i3][i4] + dmu[i2] / N_bn;
  }
}

__global__ void backwardPass_conv2d_Weight_l136(float input[64][1024][7][7], float d_output[64][2048][7][7], float d_weight[2048][1024][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 2048L*1024*1*1;  // K * C * R * S

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 2048);   // k
    const long i2 = ((idx /= 2048) % 1024);   // c
    const long i3 = ((idx /= 1024) % 1);   // r
    const long i4 = ((idx /= 1) % 1);   // s

    d_weight[i1][i2][i3][i4] = 0;
  }

  __syncthreads();


  const long N = 64L*1*1*1024*2048*7*7;  // N * R*S * C * K * P*Q

  for(long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 1); //r
    const long i3 = ((idx /= 1  ) % 1); //s
    const long i4 = ((idx /= 1  ) % 1024); //c
    const long i5 = ((idx /= 1024  ) % 2048); //k
    const long i6 = ((idx /= 2048  ) % 7); //p
    const long i7 = ((idx /= 7  ) % 7); //q

    const long h = i6 * 1 - 0 + i2;  // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3;  // w = q * stride - pad + s;

    if(h >= 0 && h < 7 && w >= 0 && w < 7)
    {
      atomicAdd(&d_weight[i5][i4][i2][i3], d_output[i1][i5][i6][i7] * input[i1][i4][h][w]);
    }
  }
}

__global__ void backwardPass_conv2d_Input_l136(float d_input[64][1024][7][7], float d_output[64][2048][7][7], float weight[2048][1024][1][1])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z = 64L*1024*7*7;   // N * C * H * W

  for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1   ) % 64);  //n
    const long i2 = ((idx /= 64   ) % 1024);  //c
    const long i3 = ((idx /= 1024   ) % 7);  //h
    const long i4 = ((idx /= 7   ) % 7);  //w

    d_input[i1][i2][i3][i4] = 0;
  }

  __syncthreads();

  const long N = 64L*1*1*1024*2048*7*7;   // N * R*S * C * K * P*Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1    ) % 64);   //n
    const long i2 = ((idx /= 64    ) % 1);   //r
    const long i3 = ((idx /= 1    ) % 1);   //s
    const long i4 = ((idx /= 1    ) % 1024);   //c
    const long i5 = ((idx /= 1024    ) % 2048);   //k
    const long i6 = ((idx /= 2048    ) % 7);   //p
    const long i7 = ((idx /= 7    ) % 7);   //q

    const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
    const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

    if (h >= 0 && h < 7 && w >= 0 && w < 7)
    {
      atomicAdd(&d_input[i1][i4][h][w], weight[i5][i4][i2][i3] * d_output[i1][i5][i6][i7]);
    }
  }
}

