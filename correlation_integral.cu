#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <iostream>

__global__ void ci_manhattan (const float *data, unsigned int *cd, const float r, const unsigned int data_len, const unsigned int dims) {

  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int b = blockIdx.x + blockIdx.y * gridDim.x;
  unsigned int i = threadIdx.x + threadIdx.y * blockDim.x;
  unsigned int t = blockDim.x*blockDim.y;

  extern __shared__ unsigned int cd_reduction[];
  cd_reduction[i] = 0;
  unsigned int cd_local = 0;
  const float *data_x = &data[x];
  for (unsigned int y = blockDim.x*blockIdx.x + 1; y < data_len; y++) {
    const float *data_y = &data[y];
    float dist = 0.f;
    for (unsigned int j = 0; j < dims; j++) {
      dist = dist + fabsf(data_x[j] - data_y[j]);
    }
    if (y > x) {
      if (r > dist) {
        cd_local += 1;
      }
    }
  }

  cd_reduction[i] = cd_local;
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
  unsigned int dim = 15;
  float a=1.4;
  float b=0.3;
  float *x, *y;
  float *d_data;
  unsigned int * d_cd;
  unsigned int * cd;
  unsigned int size = 100000;

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

  checkCudaErrors(cudaMallocHost((void **)&d_data, (nThreads*blocksPerGrid+dim)*sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_data,x,size*sizeof(float),cudaMemcpyHostToDevice));



  checkCudaErrors(cudaMallocHost((void **)&d_cd, blocksPerGrid*sizeof(unsigned int)));
  cd  =  (unsigned int *)calloc(blocksPerGrid,sizeof(unsigned int));
  cd[0] =0;
  checkCudaErrors(cudaMemcpy(d_cd,cd,blocksPerGrid*sizeof(unsigned int),cudaMemcpyHostToDevice));
  float r = pow(10.,0.5);
  std::cout << r << " " << dim << std::endl;

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  ci_manhattan<<<numBlocks, threadsPerBlock,nThreads*sizeof(unsigned int)>>>(d_data, d_cd, r, size-dim ,dim);
  cudaEventRecord(stop);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(cd,d_cd,blocksPerGrid*sizeof(unsigned int),cudaMemcpyDeviceToHost));
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Elapsed time: " << milliseconds << " miliseconds" << std::endl;

  unsigned int all = 0;
  for (unsigned int i = 0 ; i < blocksPerGrid ; i++) {
    all = all + cd[i];
  }
  std::cout << all << std::endl;	
  std::cout << all /((size - dim ) * (size - dim -1.) / 2.) << std::endl;
  
}
