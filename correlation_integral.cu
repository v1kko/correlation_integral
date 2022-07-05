#define DIMS 5
#define DATA_LEN 10000
__global__ void ci_manhattan (const float *data, unsigned int *cd, const float r) {

  const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;

  unsigned int cd_local = 0;
  float data_x[DIMS];
  float data_y[DIMS];
  #pragma unroll
  for (unsigned int i = 0; i < DIMS; i++) {
    data_x[i] = data[x+i];
  }
  for (unsigned int y = x + 1; y < DATA_LEN; y++) {
    #pragma unroll
    for (unsigned int i = 0; i < DIMS; i++) {
      data_y[i] = data[y+i];
    }
    float dist = 0.f;
    #pragma unroll
    for (unsigned int i = 0; i < DIMS; i++) {
      dist = dist + fabsf(data_x[i] - data_y[i]);
    }

    if (r > dist) {
      cd_local += 1;
    }
  }

  cd[x] = cd_local;
}
