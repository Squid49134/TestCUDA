
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <iostream>
#include <chrono>

using namespace std;

// adds elements of array in place like this for a 11 element array:
//   [1][1][1][1][1][1][1][1][1][1][1][0][0][0][0][0]
//    ^+=^  ^+=^  ^+=^  ^+=^  ^+=^  ^+=^  ^+=^  ^+=^ 
//   [2][1][2][1][2][1][2][1][2][1][1][0][0][0][0][0]
//    ^ +=  ^     ^ +=  ^     ^ +=  ^     ^ +=  ^       
//   [4][1][2][1][4][1][2][1][3][1][1][0][0][0][0][0]
//    ^    +=     ^           ^    +=     ^                   
//   [8][1][2][1][4][1][2][1][3][1][1][0][0][0][0][0]
//    ^          +=           ^                                           
//   [11][1][2][1][4][1][2][1][3][1][1][0][0][0][0][0]
//    ^ this is the final total
__global__ void addKernel(unsigned int *a, unsigned int interval, unsigned int xDim) {
    unsigned int xInd = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yInd = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i = interval * (xInd + xDim * yInd);
    a[i] = a[i] + a[i + interval / 2];
}

// Helper function for using CUDA to add vectors in parallel.
unsigned int addWithCuda(unsigned int* aHost, unsigned int size) {
    unsigned int* aDevice;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaSetDevice(0);

    // Allocate GPU buffers for the vector, ensuring that this buffer is a multiple of 2 equal to or larger than the size of the input array.  set any additional elements to 0
    unsigned int multTwoSize = 2;
    while (multTwoSize < size) {
        multTwoSize = multTwoSize * 2;
    }
    cudaMalloc(&aDevice, multTwoSize * sizeof(unsigned int));
    if (multTwoSize > size) {
        cudaMemset(&(aDevice[size]), 0, (multTwoSize - size) * sizeof(unsigned int));
    }

    // Copy input vector from host memory to GPU buffer
    cudaMemcpy(aDevice, aHost, size * sizeof(unsigned int), cudaMemcpyHostToDevice);

    auto start = chrono::high_resolution_clock::now();
    // Launch a kernel on the GPU with one thread first for every other element then every fourth and so on, synchronizing threads after each iteration
    unsigned int interval = 2;
    while (interval <= multTwoSize) {
        unsigned int numThreads = multTwoSize / interval;
        dim3 gridDim(1, 1);
        dim3 blockDim(1, 1);
        // max block dimension is 32x32 threads since the max threads per block is 1024
        // max grid dimension is 2048x2048 blocks assuming each block is 32x32 threads.  This stems from the max x and y dimensions of 65536x65536
        if (numThreads > 32) {
            blockDim.x = 32;
            if ((numThreads / 32) > 32) {
                blockDim.y = 32;
                if ((numThreads / (32 * 32)) > 2048) {
                    gridDim.x = 2048;
                    if ((numThreads / (32 * 32 * 2048)) > 2048) {
                        cout << "Array is too large" << endl;
                        return 0;
                    }
                    else {
                        gridDim.y = numThreads / (32 * 32 * 2048);
                    }
                }
                else {
                    gridDim.x = numThreads / (32 * 32);
                }
            }
            else {
                blockDim.y = numThreads / 32;
            }
        }
        else {
            blockDim.x = numThreads;
        }
        unsigned int xDim = gridDim.x * blockDim.x;
        addKernel <<< gridDim, blockDim >>> (aDevice, interval, xDim);
        cudaDeviceSynchronize();
        interval = interval * 2;
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Time:  " << duration.count() << " ms" << endl;

    // check for errors during kernel creation
    cudaError status;
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(aHost, aDevice, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // free the memory
    cudaFree(aDevice);

    // return the total
    unsigned int total = aHost[0];
    return total;
}

int main() {
    vector<unsigned int> a;
    // breaks at 536870913 since this is 2^29 + 1 so multTwo array length will be rounded up to 2^30 = 1073741824 integers times 4 bytes per integer is 4GB which is all of the available GPU memory
    for (unsigned int i = 0; i < 536870912; i++) {
        a.push_back(1);
    }

    // Add elements of the vector in parallel.
    unsigned int total = addWithCuda(&(a[0]), a.size());

    cout << total << endl;

    // cudaDeviceReset must be called before exiting in order for profiling and tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();

    return 0;
}
