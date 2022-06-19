#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>

__device__ float* cd_p;

__global__ void ci_manhattan (const float *data, const float *r, int data_len, int r_len, int dims) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  float dist = 0;
  if (x < data_len) {
    if (y < data_len && y > x) {
      for (int i = 0; i < dims ; i++) {
        dist = dist + fabsf(data[x+i] + data[y+i]);
      }
      for (int i = 0; i < r_len ; i++) {
        if (r[i] > dist) {
          cd_p[i*data_len+ x] = cd_p[i*data_len + x] + 1;
        }
      }
    }
  }
}


int main(void) {
  float a=1.4;
  float b=0.3;
  float *x, *y;
  float *d_data;
  float *d_r;
  float rs[5] = {-2.5,-2,-1,-.5,0.5};


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


  checkCudaErrors(cudaMallocHost((void **)&cd_p, 5*10000*sizeof(float)));
  checkCudaErrors(cudaMallocHost((void **)&d_data, 10000*sizeof(float)));
  checkCudaErrors(cudaMallocHost((void **)&d_r, 5*sizeof(float)));

  checkCudaErrors(cudaMemcpy(d_data,x,10000*sizeof(float),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_r,rs,5*sizeof(float),cudaMemcpyHostToDevice));

  int threadsPerBlock = 256;
  int blocksPerGrid = (10000 + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  ci_manhattan<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_r, 10000-5, 5,5);
  checkCudaErrors(cudaGetLastError());
}
