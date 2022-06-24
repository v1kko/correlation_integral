#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <iostream>



__global__ void ci_manhattan (const float *data, unsigned int *cd, const float r, unsigned int data_len, unsigned int dims) {
  if (blockIdx.x > blockIdx.y) { return; }
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int b = blockIdx.x + blockIdx.y * gridDim.x;
  unsigned int i = threadIdx.x + threadIdx.y * blockDim.x;
  unsigned int t = blockDim.x*blockDim.y;

  extern __shared__ char shared[];

  unsigned int * cd_reduction = (unsigned int *) shared;
  float * data_x              = (float*) &shared[t*sizeof(unsigned int)];
  float * data_y              = (float*) &shared[t*sizeof(unsigned int)+(blockDim.x+dims)*sizeof(float)];
  cd_reduction[i] = 0;

  //Index is shifted by one! 
  if (i < blockDim.x + dims) {
    if ( blockDim.x*blockIdx.x+i+1 < data_len+dims ) {
      data_x[i] = data[blockDim.x*blockIdx.x+i+1];
    }
  }
  __syncthreads();
  if (i < blockDim.x + dims) {
    if ( blockDim.y*blockIdx.y+i+1 < data_len+dims ) {
      data_y[i] = data[blockDim.y*blockIdx.y+i+1];
    }
  }
  __syncthreads();

  if (x < data_len && y < data_len && y > x) {
    float dist = 0.f;
    //j == 0 is next neighbour because data_x/y array shifted by one
    for (unsigned int j = 0; j < dims ; j++) {
      dist = dist + fabsf(data_x[threadIdx.x+j] - data_y[threadIdx.y+j]);
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
  unsigned int dim = 5;
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


  checkCudaErrors(cudaMallocHost((void **)&d_data, size*sizeof(float)));

  checkCudaErrors(cudaMemcpy(d_data,x,size*sizeof(float),cudaMemcpyHostToDevice));

  dim3 threadsPerBlock(32,32);
  unsigned int nThreads = threadsPerBlock.x*threadsPerBlock.y;
  dim3 numBlocks(((size-1) / threadsPerBlock.x)+1, ((size-1) / threadsPerBlock.y)+1);
  unsigned int blocksPerGrid = numBlocks.x * numBlocks.y;

  checkCudaErrors(cudaMallocHost((void **)&d_cd, blocksPerGrid*sizeof(unsigned int)));
  cd  =  (unsigned int *)calloc(blocksPerGrid,sizeof(unsigned int));
  cd[0] =0;
  checkCudaErrors(cudaMemcpy(d_cd,cd,blocksPerGrid*sizeof(unsigned int),cudaMemcpyHostToDevice));

  printf("CUDA kernel launch with %u blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock.x*threadsPerBlock.y);
  float r = pow(10.,0.5);
  std::cout << r << " " << dim << std::endl;
  ci_manhattan<<<numBlocks, threadsPerBlock,nThreads*sizeof(unsigned int)+(threadsPerBlock.x+dim)*2*sizeof(float)>>>(d_data, d_cd, r, size-dim ,dim);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(cd,d_cd,blocksPerGrid*sizeof(unsigned int),cudaMemcpyDeviceToHost));

  unsigned int all = 0;
  for (unsigned int i = 0 ; i < blocksPerGrid ; i++) {
    all = all + cd[i];
  }
  std::cout << all << std::endl;	
  std::cout << all /((size - dim ) * (size - dim -1.) / 2.) << std::endl;
  
}
