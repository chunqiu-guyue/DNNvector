__global__ void forwardPass_conv2d_l0(float input[64][3][224][224], float output[64][64][112][112], float weight[64][3][7][7])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*64*112*112;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 64);     // k
    const long i3 = ((idx /= 64) % 112);     // p
    const long i4 = ((idx /= 112) % 112);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*7*7*3*64*112*112;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 7);     // r
		const long i3 = ((idx /= 7) % 7);     // s
		const long i4 = ((idx /= 7) % 3);     // c
		const long i5 = ((idx /= 3) % 64);     // k
		const long i6 = ((idx /= 64) % 112);     // p
		const long i7 = ((idx /= 112) % 112);     // q
		const long h = i6 * 2 - 3 + i2; // h = p * stride - pad + r;
		const long w = i7 * 2 - 3 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 224 && w >= 0 && w < 224)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l1(float input[64][64][112][112], float gamma_and_beta[128], float running_m[64], float running_v[64], float mu[64], float var[64], float v1[64][64][112][112], float v2[64], float output[64][64][112][112]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 64L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*64*112*112;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 64); // c
    const long i3 = ((idx /= 64) % 112); // h
    const long i4 = ((idx /= 112) % 112); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*112*112;  // N * H * W
  const long N2 = 64; // C
  
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
    const long i2 = ((idx /= 64) % 64); // c
    const long i3 = ((idx /= 64) % 112); // h
    const long i4 = ((idx /= 112) % 112); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 64); // c
    const long i3 = ((idx /= 64) % 112); // h
    const long i4 = ((idx /= 112) % 112); // w

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
    const long i1 = ((idx /= 1) % 64);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 64); // c
    const long i3 = ((idx /= 64) % 112); // h
    const long i4 = ((idx /= 112) % 112); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[64 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_ReLU_l2(float input[64][64][112][112], float output[64][64][112][112]){

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

    output[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? input[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void forwardPass_MaxPool2d_l3(float input[64][64][112][112], float output[64][64][56][56])
{
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long N = 64L*64*56*56; // N * K * P * Q

  for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1  ) % 64); //n
    const long i2 = ((idx /= 64  ) % 64); //k
    const long i3 = ((idx /= 64  ) % 56); //p
    const long i4 = ((idx /= 56  ) % 56); //q

    float max = -9999999;

    for (long k0 = 0; k0 < 3; k0++)  //kernel_size0
    {
      for (long k1 = 0; k1 < 3; k1++) //kernelsize_1
      {
        const long h = i3 * 2 - 1 + k0; // h = p * stride - pad + k0;
        const long w = i4 * 2 - 1 + k1; // w = q * stride - pad + k1;
        if (h >= 0 && h < 112 && w >= 0 && w < 112)
        {
          max = max > input[i1][i2][h][w] ? max : input[i1][i2][h][w];
        }
      }
    }
    output[i1][i2][i3][i4] = max;
  }
}

__global__ void forwardPass_conv2d_l4(float input[64][64][56][56], float output[64][128][56][56], float weight[128][64][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*128*56*56;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 128);     // k
    const long i3 = ((idx /= 128) % 56);     // p
    const long i4 = ((idx /= 56) % 56);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*64*128*56*56;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 64);     // c
		const long i5 = ((idx /= 64) % 128);     // k
		const long i6 = ((idx /= 128) % 56);     // p
		const long i7 = ((idx /= 56) % 56);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 56 && w >= 0 && w < 56)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l5(float input[64][128][56][56], float gamma_and_beta[256], float running_m[128], float running_v[128], float mu[128], float var[128], float v1[64][128][56][56], float v2[128], float output[64][128][56][56]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 128L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*128*56*56;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*56*56;  // N * H * W
  const long N2 = 128; // C
  
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
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

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
    const long i1 = ((idx /= 1) % 128);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[128 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l6(float input[64][128][56][56], float output[64][128][56][56], float weight[128][128][3][3])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*128*56*56;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 128);     // k
    const long i3 = ((idx /= 128) % 56);     // p
    const long i4 = ((idx /= 56) % 56);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*3*3*128*128*56*56;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 3);     // r
		const long i3 = ((idx /= 3) % 3);     // s
		const long i4 = ((idx /= 3) % 128);     // c
		const long i5 = ((idx /= 128) % 128);     // k
		const long i6 = ((idx /= 128) % 56);     // p
		const long i7 = ((idx /= 56) % 56);     // q
		const long h = i6 * 1 - 1 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 1 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 56 && w >= 0 && w < 56)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l7(float input[64][128][56][56], float gamma_and_beta[256], float running_m[128], float running_v[128], float mu[128], float var[128], float v1[64][128][56][56], float v2[128], float output[64][128][56][56]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 128L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*128*56*56;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*56*56;  // N * H * W
  const long N2 = 128; // C
  
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
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

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
    const long i1 = ((idx /= 1) % 128);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[128 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l8(float input[64][128][56][56], float output[64][256][56][56], float weight[256][128][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*256*56*56;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 256);     // k
    const long i3 = ((idx /= 256) % 56);     // p
    const long i4 = ((idx /= 56) % 56);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*128*256*56*56;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 128);     // c
		const long i5 = ((idx /= 128) % 256);     // k
		const long i6 = ((idx /= 256) % 56);     // p
		const long i7 = ((idx /= 56) % 56);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 56 && w >= 0 && w < 56)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l9(float input[64][256][56][56], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][56][56], float v2[256], float output[64][256][56][56]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 256L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*256*56*56;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*56*56;  // N * H * W
  const long N2 = 256; // C
  
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
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

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
    const long i1 = ((idx /= 1) % 256);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[256 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_ReLU_l10(float input[64][256][56][56], float output[64][256][56][56]){

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

    output[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? input[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void forwardPass_conv2d_l11(float input[64][64][56][56], float output[64][256][56][56], float weight[256][64][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*256*56*56;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 256);     // k
    const long i3 = ((idx /= 256) % 56);     // p
    const long i4 = ((idx /= 56) % 56);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*64*256*56*56;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 64);     // c
		const long i5 = ((idx /= 64) % 256);     // k
		const long i6 = ((idx /= 256) % 56);     // p
		const long i7 = ((idx /= 56) % 56);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 56 && w >= 0 && w < 56)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l12(float input[64][256][56][56], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][56][56], float v2[256], float output[64][256][56][56]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 256L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*256*56*56;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*56*56;  // N * H * W
  const long N2 = 256; // C
  
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
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

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
    const long i1 = ((idx /= 1) % 256);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[256 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_Add_l13(float* input0, float* input1, float* out, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        out[idx] = input0[idx] + input1[idx];
      }
  }
}


__global__ void forwardPass_conv2d_l14(float input[64][256][56][56], float output[64][128][56][56], float weight[128][256][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*128*56*56;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 128);     // k
    const long i3 = ((idx /= 128) % 56);     // p
    const long i4 = ((idx /= 56) % 56);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*256*128*56*56;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 256);     // c
		const long i5 = ((idx /= 256) % 128);     // k
		const long i6 = ((idx /= 128) % 56);     // p
		const long i7 = ((idx /= 56) % 56);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 56 && w >= 0 && w < 56)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l15(float input[64][128][56][56], float gamma_and_beta[256], float running_m[128], float running_v[128], float mu[128], float var[128], float v1[64][128][56][56], float v2[128], float output[64][128][56][56]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 128L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*128*56*56;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*56*56;  // N * H * W
  const long N2 = 128; // C
  
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
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

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
    const long i1 = ((idx /= 1) % 128);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[128 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l16(float input[64][128][56][56], float output[64][128][56][56], float weight[128][128][3][3])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*128*56*56;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 128);     // k
    const long i3 = ((idx /= 128) % 56);     // p
    const long i4 = ((idx /= 56) % 56);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*3*3*128*128*56*56;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 3);     // r
		const long i3 = ((idx /= 3) % 3);     // s
		const long i4 = ((idx /= 3) % 128);     // c
		const long i5 = ((idx /= 128) % 128);     // k
		const long i6 = ((idx /= 128) % 56);     // p
		const long i7 = ((idx /= 56) % 56);     // q
		const long h = i6 * 1 - 1 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 1 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 56 && w >= 0 && w < 56)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l17(float input[64][128][56][56], float gamma_and_beta[256], float running_m[128], float running_v[128], float mu[128], float var[128], float v1[64][128][56][56], float v2[128], float output[64][128][56][56]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 128L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*128*56*56;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*56*56;  // N * H * W
  const long N2 = 128; // C
  
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
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

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
    const long i1 = ((idx /= 1) % 128);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[128 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l18(float input[64][128][56][56], float output[64][256][56][56], float weight[256][128][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*256*56*56;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 256);     // k
    const long i3 = ((idx /= 256) % 56);     // p
    const long i4 = ((idx /= 56) % 56);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*128*256*56*56;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 128);     // c
		const long i5 = ((idx /= 128) % 256);     // k
		const long i6 = ((idx /= 256) % 56);     // p
		const long i7 = ((idx /= 56) % 56);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 56 && w >= 0 && w < 56)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l19(float input[64][256][56][56], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][56][56], float v2[256], float output[64][256][56][56]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 256L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*256*56*56;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*56*56;  // N * H * W
  const long N2 = 256; // C
  
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
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

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
    const long i1 = ((idx /= 1) % 256);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[256 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_ReLU_l20(float input[64][256][56][56], float output[64][256][56][56]){

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

    output[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? input[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void forwardPass_Add_l21(float* input0, float* input1, float* out, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        out[idx] = input0[idx] + input1[idx];
      }
  }
}


__global__ void forwardPass_conv2d_l22(float input[64][256][56][56], float output[64][128][56][56], float weight[128][256][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*128*56*56;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 128);     // k
    const long i3 = ((idx /= 128) % 56);     // p
    const long i4 = ((idx /= 56) % 56);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*256*128*56*56;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 256);     // c
		const long i5 = ((idx /= 256) % 128);     // k
		const long i6 = ((idx /= 128) % 56);     // p
		const long i7 = ((idx /= 56) % 56);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 56 && w >= 0 && w < 56)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l23(float input[64][128][56][56], float gamma_and_beta[256], float running_m[128], float running_v[128], float mu[128], float var[128], float v1[64][128][56][56], float v2[128], float output[64][128][56][56]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 128L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*128*56*56;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*56*56;  // N * H * W
  const long N2 = 128; // C
  
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
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

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
    const long i1 = ((idx /= 1) % 128);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[128 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l24(float input[64][128][56][56], float output[64][128][56][56], float weight[128][128][3][3])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*128*56*56;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 128);     // k
    const long i3 = ((idx /= 128) % 56);     // p
    const long i4 = ((idx /= 56) % 56);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*3*3*128*128*56*56;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 3);     // r
		const long i3 = ((idx /= 3) % 3);     // s
		const long i4 = ((idx /= 3) % 128);     // c
		const long i5 = ((idx /= 128) % 128);     // k
		const long i6 = ((idx /= 128) % 56);     // p
		const long i7 = ((idx /= 56) % 56);     // q
		const long h = i6 * 1 - 1 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 1 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 56 && w >= 0 && w < 56)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l25(float input[64][128][56][56], float gamma_and_beta[256], float running_m[128], float running_v[128], float mu[128], float var[128], float v1[64][128][56][56], float v2[128], float output[64][128][56][56]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 128L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 128);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*128*56*56;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*56*56;  // N * H * W
  const long N2 = 128; // C
  
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
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

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
    const long i1 = ((idx /= 1) % 128);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 128); // c
    const long i3 = ((idx /= 128) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[128 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l26(float input[64][128][56][56], float output[64][256][56][56], float weight[256][128][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*256*56*56;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 256);     // k
    const long i3 = ((idx /= 256) % 56);     // p
    const long i4 = ((idx /= 56) % 56);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*128*256*56*56;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 128);     // c
		const long i5 = ((idx /= 128) % 256);     // k
		const long i6 = ((idx /= 256) % 56);     // p
		const long i7 = ((idx /= 56) % 56);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 56 && w >= 0 && w < 56)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l27(float input[64][256][56][56], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][56][56], float v2[256], float output[64][256][56][56]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 256L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*256*56*56;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*56*56;  // N * H * W
  const long N2 = 256; // C
  
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
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

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
    const long i1 = ((idx /= 1) % 256);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[256 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_ReLU_l28(float input[64][256][56][56], float output[64][256][56][56]){

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

    output[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? input[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void forwardPass_Add_l29(float* input0, float* input1, float* out, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        out[idx] = input0[idx] + input1[idx];
      }
  }
}


__global__ void forwardPass_conv2d_l30(float input[64][256][56][56], float output[64][256][56][56], float weight[256][256][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*256*56*56;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 256);     // k
    const long i3 = ((idx /= 256) % 56);     // p
    const long i4 = ((idx /= 56) % 56);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*256*256*56*56;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 256);     // c
		const long i5 = ((idx /= 256) % 256);     // k
		const long i6 = ((idx /= 256) % 56);     // p
		const long i7 = ((idx /= 56) % 56);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 56 && w >= 0 && w < 56)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l31(float input[64][256][56][56], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][56][56], float v2[256], float output[64][256][56][56]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 256L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*256*56*56;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*56*56;  // N * H * W
  const long N2 = 256; // C
  
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
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

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
    const long i1 = ((idx /= 1) % 256);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 56); // h
    const long i4 = ((idx /= 56) % 56); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[256 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l32(float input[64][256][56][56], float output[64][256][28][28], float weight[256][256][3][3])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*256*28*28;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 256);     // k
    const long i3 = ((idx /= 256) % 28);     // p
    const long i4 = ((idx /= 28) % 28);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*3*3*256*256*28*28;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 3);     // r
		const long i3 = ((idx /= 3) % 3);     // s
		const long i4 = ((idx /= 3) % 256);     // c
		const long i5 = ((idx /= 256) % 256);     // k
		const long i6 = ((idx /= 256) % 28);     // p
		const long i7 = ((idx /= 28) % 28);     // q
		const long h = i6 * 2 - 1 + i2; // h = p * stride - pad + r;
		const long w = i7 * 2 - 1 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 56 && w >= 0 && w < 56)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l33(float input[64][256][28][28], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][28][28], float v2[256], float output[64][256][28][28]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 256L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*256*28*28;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*28*28;  // N * H * W
  const long N2 = 256; // C
  
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
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

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
    const long i1 = ((idx /= 1) % 256);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[256 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l34(float input[64][256][28][28], float output[64][512][28][28], float weight[512][256][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*512*28*28;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 512);     // k
    const long i3 = ((idx /= 512) % 28);     // p
    const long i4 = ((idx /= 28) % 28);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*256*512*28*28;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 256);     // c
		const long i5 = ((idx /= 256) % 512);     // k
		const long i6 = ((idx /= 512) % 28);     // p
		const long i7 = ((idx /= 28) % 28);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 28 && w >= 0 && w < 28)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l35(float input[64][512][28][28], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][28][28], float v2[512], float output[64][512][28][28]) { 
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

  const long N1 = 64L*512*28*28;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*28*28;  // N * H * W
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
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

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
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[512 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_ReLU_l36(float input[64][512][28][28], float output[64][512][28][28]){

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

    output[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? input[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void forwardPass_conv2d_l37(float input[64][256][56][56], float output[64][512][28][28], float weight[512][256][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*512*28*28;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 512);     // k
    const long i3 = ((idx /= 512) % 28);     // p
    const long i4 = ((idx /= 28) % 28);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*256*512*28*28;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 256);     // c
		const long i5 = ((idx /= 256) % 512);     // k
		const long i6 = ((idx /= 512) % 28);     // p
		const long i7 = ((idx /= 28) % 28);     // q
		const long h = i6 * 2 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 2 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 56 && w >= 0 && w < 56)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l38(float input[64][512][28][28], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][28][28], float v2[512], float output[64][512][28][28]) { 
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

  const long N1 = 64L*512*28*28;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*28*28;  // N * H * W
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
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

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
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[512 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_Add_l39(float* input0, float* input1, float* out, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        out[idx] = input0[idx] + input1[idx];
      }
  }
}


__global__ void forwardPass_conv2d_l40(float input[64][512][28][28], float output[64][256][28][28], float weight[256][512][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*256*28*28;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 256);     // k
    const long i3 = ((idx /= 256) % 28);     // p
    const long i4 = ((idx /= 28) % 28);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*512*256*28*28;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 512);     // c
		const long i5 = ((idx /= 512) % 256);     // k
		const long i6 = ((idx /= 256) % 28);     // p
		const long i7 = ((idx /= 28) % 28);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 28 && w >= 0 && w < 28)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l41(float input[64][256][28][28], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][28][28], float v2[256], float output[64][256][28][28]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 256L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*256*28*28;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*28*28;  // N * H * W
  const long N2 = 256; // C
  
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
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

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
    const long i1 = ((idx /= 1) % 256);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[256 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l42(float input[64][256][28][28], float output[64][256][28][28], float weight[256][256][3][3])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*256*28*28;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 256);     // k
    const long i3 = ((idx /= 256) % 28);     // p
    const long i4 = ((idx /= 28) % 28);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*3*3*256*256*28*28;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 3);     // r
		const long i3 = ((idx /= 3) % 3);     // s
		const long i4 = ((idx /= 3) % 256);     // c
		const long i5 = ((idx /= 256) % 256);     // k
		const long i6 = ((idx /= 256) % 28);     // p
		const long i7 = ((idx /= 28) % 28);     // q
		const long h = i6 * 1 - 1 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 1 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 28 && w >= 0 && w < 28)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l43(float input[64][256][28][28], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][28][28], float v2[256], float output[64][256][28][28]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 256L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*256*28*28;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*28*28;  // N * H * W
  const long N2 = 256; // C
  
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
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

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
    const long i1 = ((idx /= 1) % 256);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[256 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_conv2d_l44(float input[64][256][28][28], float output[64][512][28][28], float weight[512][256][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*512*28*28;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 512);     // k
    const long i3 = ((idx /= 512) % 28);     // p
    const long i4 = ((idx /= 28) % 28);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*256*512*28*28;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 256);     // c
		const long i5 = ((idx /= 256) % 512);     // k
		const long i6 = ((idx /= 512) % 28);     // p
		const long i7 = ((idx /= 28) % 28);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 28 && w >= 0 && w < 28)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l45(float input[64][512][28][28], float gamma_and_beta[1024], float running_m[512], float running_v[512], float mu[512], float var[512], float v1[64][512][28][28], float v2[512], float output[64][512][28][28]) { 
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

  const long N1 = 64L*512*28*28;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*28*28;  // N * H * W
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
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 512); // c
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

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
    const long i3 = ((idx /= 512) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[512 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

__global__ void forwardPass_ReLU_l46(float input[64][512][28][28], float output[64][512][28][28]){

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

    output[i1][i2][i3][i4] = input[i1][i2][i3][i4] >= 0 ? input[i1][i2][i3][i4] : 0 ;
  }
}

__global__ void forwardPass_Add_l47(float* input0, float* input1, float* out, long N) {
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  for(long n = 0; n < N; n += parallel_size) {
    long idx = n + thread_pos;
      if (idx < N) {
        out[idx] = input0[idx] + input1[idx];
      }
  }
}


__global__ void forwardPass_conv2d_l48(float input[64][512][28][28], float output[64][256][28][28], float weight[256][512][1][1])
{
	const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
	const long parallel_size = blockDim.x * gridDim.x;
	const long Z = 64L*256*28*28;     //N * K * P*Q 

	for (long n = Z * thread_pos / parallel_size; n < Z * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64);     // n
    const long i2 = ((idx /= 64) % 256);     // k
    const long i3 = ((idx /= 256) % 28);     // p
    const long i4 = ((idx /= 28) % 28);     // q
    output[i1][i2][i3][i4] = 0;
	}
	__syncthreads();
	const long N = 64L*1*1*512*256*28*28;     //N * R*S * C * K * P*Q 

	for (long n = N * thread_pos / parallel_size; n < N * (thread_pos+1) / parallel_size; ++n) {
		long idx = n;
		const long i1 = ((idx /= 1) % 64);     // n
		const long i2 = ((idx /= 64) % 1);     // r
		const long i3 = ((idx /= 1) % 1);     // s
		const long i4 = ((idx /= 1) % 512);     // c
		const long i5 = ((idx /= 512) % 256);     // k
		const long i6 = ((idx /= 256) % 28);     // p
		const long i7 = ((idx /= 28) % 28);     // q
		const long h = i6 * 1 - 0 + i2; // h = p * stride - pad + r;
		const long w = i7 * 1 - 0 + i3; // w = q * stride - pad + s;

		if (h >= 0 && h < 28 && w >= 0 && w < 28)
		{
 			atomicAdd(&output[i1][i5][i6][i7], weight[i5][i4][i2][i3] * input[i1][i4][h][w]);
		}
	}
}

__global__ void forwardPass_BatchNorm2d_l49(float input[64][256][28][28], float gamma_and_beta[512], float running_m[256], float running_v[256], float mu[256], float var[256], float v1[64][256][28][28], float v2[256], float output[64][256][28][28]) { 
  const long thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
  const long parallel_size = blockDim.x * gridDim.x;

  const long Z1 = 256L;  //C

  for (long n = Z1 * thread_pos / parallel_size; n < Z1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 256);  // c

    mu[i1] = 0;
    var[i1] = 0;
  }

  __syncthreads();

  const long N1 = 64L*256*28*28;  // N * C * H * W

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    atomicAdd(&mu[i2], input[i1][i2][i3][i4]);
  }

  __syncthreads();


  const long N_bn = 64L*28*28;  // N * H * W
  const long N2 = 256; // C
  
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
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    v1[i1][i2][i3][i4] = input[i1][i2][i3][i4] - running_m[i2];

  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

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
    const long i1 = ((idx /= 1) % 256);  // c

    v2[i1] = sqrt(running_v[i1] + 0.000010); //eps = 0.00001, v2 = sqrt(running_v + eps)
  }

  __syncthreads();

  for (long n = N1 * thread_pos / parallel_size; n < N1 * (thread_pos+1) / parallel_size; ++n) {
    long idx = n;
    const long i1 = ((idx /= 1) % 64); // n
    const long i2 = ((idx /= 64) % 256); // c
    const long i3 = ((idx /= 256) % 28); // h
    const long i4 = ((idx /= 28) % 28); // w

    output[i1][i2][i3][i4] = (v1[i1][i2][i3][i4] / v2[i2]) * gamma_and_beta[i2] + gamma_and_beta[256 + i2]; //gamma = gamma_and_beta[0~c-1], beta = gamma_and_beta[c~2c-1]; output = (v1/v2) * gamma + beta
  }
}

