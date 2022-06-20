#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <iostream>

__global__ void ci_manhattan (const float *data, int *cd, const float r, int data_len, int r_len, int dims) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int b = blockIdx.x + blockIdx.y * blockDim.x;
  int i = threadIdx.x + threadIdx.y * blockDim.x;
  int t = blockDim.x*blockDim.y;
  float dist = 0.f;
  extern __shared__ int cd_reduction[];
  cd_reduction[i] = 0;
  if (x < data_len && y < data_len && y > x) {
    for (int j = 0; j < dims ; j++) {
      dist = dist + fabsf(data[x+j] + data[y+j]);
    }
    if (r > dist) {
      cd_reduction[i] = 1;
    }
  }  
   __syncthreads();
  for (unsigned int s = t / 2; s > 0; s >>= 1) {
    if (i < s) {
      cd_reduction[i] += cd_reduction[i + s];
    }
  }
  if (i == 0) {
    cd[b] = cd_reduction[0];
  }
}


int main(void) {
  float a=1.4;
  float b=0.3;
  float *x, *y;
  float *d_data;
  int * d_cd;
  int * cd;

  x = (float*)malloc(15000*sizeof(float));
  y = (float*)malloc(15000*sizeof(float));
  
  x[0]=1;
  y[0]=0;
  for (int i = 1; i < 15000-1; i++) {
      x[i+1]=1-a*x[i]*x[i]+y[i];
      y[i+1]=b*x[i];
  }

  x=&x[5000];
  y=&y[5000];


  checkCudaErrors(cudaMallocHost((void **)&d_data, 10000*sizeof(float)));

  checkCudaErrors(cudaMemcpy(d_data,x,10000*sizeof(float),cudaMemcpyHostToDevice));

  int threadsPerBlock = 256;
  int blocksPerGrid = (10000 + threadsPerBlock - 1) / threadsPerBlock;

  checkCudaErrors(cudaMallocHost((void **)&d_cd, blocksPerGrid*sizeof(int)));
  cd  =  (int *)malloc(blocksPerGrid*sizeof(int));

  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  ci_manhattan<<<blocksPerGrid, threadsPerBlock,threadsPerBlock*sizeof(int)>>>(d_data, d_cd, 0.5, 10000-5, 5,5);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(cd,d_cd,blocksPerGrid*sizeof(int),cudaMemcpyDeviceToHost));

  for (int i = 0 ; i < blocksPerGrid ; i++) {
    std::cout << cd[i] << " ";
  }
}
