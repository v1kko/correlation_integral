#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <iostream>

#define DIMS 5
__global__ void ci_manhattan (const float *data, unsigned int *cd, const float r, const unsigned int data_len) {

  const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;

  unsigned int cd_local = 0;
  float data_x[DIMS];
  #pragma unroll
  for (unsigned int i = 0; i < DIMS; i++) {
    data_x[i] = data[x+i];
  }
  for (unsigned int y = blockDim.x*blockIdx.x + 1; y < data_len; y++) {
    const float *data_y = &data[y];
    float dist = 0.f;
    #pragma unroll
    for (unsigned int j = 0; j < DIMS; j++) {
      dist = dist + fabsf(data_x[j] - data_y[j]);
    }
    if (y > x) {
      if (r > dist) {
        cd_local += 1;
      }
    }
  }

  cd[x] = cd_local;
}

int main(void) {
  float a=1.4;
  float b=0.3;
  float *x, *y;
  float *d_data;
  unsigned int * d_cd;
  unsigned int * cd;
  unsigned int size = 100000;
  unsigned int data_len = size - DIMS;

  x = (float*)malloc((size +5000)*sizeof(float));
  y = (float*)malloc((size +5000)*sizeof(float));
  
  x[0]=1;
  y[0]=0;
  for (unsigned int i = 0; i < size + 5000-1; i++) {
      x[i+1]=1-a*x[i]*x[i]+y[i];
      y[i+1]=b*x[i];
  }

  x=&x[5000];
  y=&y[5000];


  dim3 threadsPerBlock(1024);
  unsigned int nThreads = threadsPerBlock.x;
  dim3 numBlocks(((size-1) / threadsPerBlock.x)+1);
  unsigned int blocksPerGrid = numBlocks.x;

  checkCudaErrors(cudaMallocHost((void **)&d_data, (nThreads*blocksPerGrid+DIMS)*sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_data,x,size*sizeof(float),cudaMemcpyHostToDevice));



  checkCudaErrors(cudaMallocHost((void **)&d_cd,(nThreads*blocksPerGrid)*sizeof(unsigned int)));
  cd  =  (unsigned int *)malloc(nThreads*blocksPerGrid*sizeof(unsigned int));
  float r = pow(10.,0.5);
  std::cout << r << " " << DIMS << std::endl;

  ci_manhattan<<<numBlocks, threadsPerBlock>>>(d_data, d_cd, r, data_len);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(cd,d_cd,nThreads*blocksPerGrid*sizeof(unsigned int),cudaMemcpyDeviceToHost));

  unsigned int all = 0;
  for (unsigned int i = 0 ; i < data_len-1 ; i++) {
    all = all + cd[i];
  }
  std::cout << all << std::endl;	
  std::cout << all /((data_len ) * (data_len -1.) / 2.) << std::endl;
  
}
