#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <iostream>

__global__ void ci_manhattan (const float *data, unsigned int *cd, const float r, unsigned int data_len, unsigned int dims) {
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int b = blockIdx.x + blockIdx.y * blockDim.x;
  unsigned int i = threadIdx.x + threadIdx.y * blockDim.x;
  unsigned int t = blockDim.x*blockDim.y;
  float dist = 0.f;
  extern __shared__ int cd_reduction[];
  cd_reduction[i] = 0;
  if (x < data_len && y < data_len && y > x) {
    for (unsigned int j = 1; j <= dims ; j++) {
      dist = dist + fabsf(data[x+j] - data[y+j]);
    }
    if (r > dist) {
      cd_reduction[i] = 1;
    }
  }  
   __syncthreads();
  for (unsigned int s = t/2; s > 0; s >>= 1) {
    if (i < s) {
      cd_reduction[i] += cd_reduction[i + s];
    }
    __syncthreads();
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
  unsigned int * d_cd;
  unsigned int * cd;
  unsigned int size = 10000;

  x = (float*)malloc(15000*sizeof(float));
  y = (float*)malloc(15000*sizeof(float));
  
  x[0]=1;
  y[0]=0;
  for (unsigned int i = 0; i < 15000-1; i++) {
      x[i+1]=1-a*x[i]*x[i]+y[i];
      y[i+1]=b*x[i];
  }

  x=&x[5000];
  y=&y[5000];


  checkCudaErrors(cudaMallocHost((void **)&d_data, size*sizeof(float)));

  checkCudaErrors(cudaMemcpy(d_data,x,size*sizeof(float),cudaMemcpyHostToDevice));

  dim3 threadsPerBlock(16,16);
  dim3 numBlocks(((size-1) / threadsPerBlock.x)+1, ((size-1) / threadsPerBlock.y)+1);
  unsigned int blocksPerGrid = numBlocks.x * numBlocks.y;

  checkCudaErrors(cudaMallocHost((void **)&d_cd, blocksPerGrid*sizeof(unsigned int)));
  cd  =  (unsigned int *)malloc(blocksPerGrid*sizeof(unsigned int));

  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock.x*threadsPerBlock.y);
  float r = pow(10.,0.5);
  std::cout << r << std::endl;
  ci_manhattan<<<numBlocks, threadsPerBlock,256*sizeof(unsigned int)>>>(d_data, d_cd, r, size-5 ,5);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(cd,d_cd,blocksPerGrid*sizeof(unsigned int),cudaMemcpyDeviceToHost));

  unsigned int all = 0;
  for (unsigned int i = 0 ; i < blocksPerGrid ; i++) {
    all = all + cd[i];
  }
	
  std::cout << all /((size - 5. ) * (size - 5. -1.) / 2.) << std::endl;
  
}
