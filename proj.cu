#include <iostream> 
#include <stdlib.h> 
#include "cuda_runtime.h"
#include <sys/time.h>
#include <float.h>
#include <fstream>
#include "opencv2/opencv.hpp"

#define cvCellSizeX 65
#define cvCellSizeY 15
#define cvFontSize 0.6

using namespace std; 


#define block 16
#define tile block // tile is a differnt define directive for better usability. 

struct timeval startMult; 
struct timeval endMult; 
struct timeval startMin; 
struct timeval endMin;
struct timeval ssspStart; 
struct timeval ssspEnd;  

dim3 threadsPerBlock(block, block);

#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}


typedef struct {
    float val; 
    int16_t row, col; //intially had an int. Exceeds bounds for shared memory. int8_t is not enough because the index can have a value upto 16000 something in our case. 
}minVal; 

typedef struct{
    float val;
    int index;  
} path; 


__device__ void warpReduce(volatile minVal *minlocal, int threadId){// the for loop in block reduce gives us the 64 smallest values if block size is 32. So 2 warps
    
    for(int x = block ; x >= 1 ; x = x/2){ // x is 32 in the first case. 
        if(minlocal[threadId].val>minlocal[threadId+x].val){ // the values are not swapped properly if the key word volatile is not used. 
        //minlocal[threadId] = minlocal[threadId+32];
        minlocal[threadId].val= minlocal[threadId+x].val;
        minlocal[threadId].row= minlocal[threadId+x].row;
        minlocal[threadId].col= minlocal[threadId+x].col;
    }
    }
}

__device__ void blockReduce(minVal *minlocal, int threadId){
    for(unsigned int s = (block*block)/2 ; s>block; s = s >> 1){//divide the stride by 2 every iteration -- parallel reduction sequential addressing- from the nvidia ppt.  
        if (threadId < s){ // if the threadid is greater than the stride value then its already been compared with another thread. 
            if(minlocal[threadId].val > minlocal[threadId + s].val){
                minlocal[threadId] = minlocal[threadId + s]; //if the threadId + stride value is smaller then we swap the value row and col else we do nothing. (we save the smaller value and don't care about the larget value) 
            }
        }
        __syncthreads(); 
    }

    if(threadId < block){
        warpReduce(minlocal, threadId);
    }
}


__global__ void matrixMult(float* arr1, float* arr2, float *result, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    int threadId = threadIdx.y * block + threadIdx.x; 

    __shared__ float s_arr1[block*block];
    __shared__ float s_arr2[block*block]; // this allows us to use the shared mem for min computation. 

    float sum = 0.0; 

    for (int i = 0; i < (size/block); i++){
        s_arr1[threadId] = arr1[row*size + (i*tile + threadIdx.x)]; 
        s_arr2[threadId] = arr2[(i*tile + threadIdx.y)*size + col]; 
        __syncthreads(); // all tiles should be loaded before the next step. 

            for(int j=0; j<tile; j++){
                sum += s_arr1[threadIdx.y * tile + j] * s_arr2[j*tile +threadIdx.x]; 
            }

            __syncthreads();

    }
    

    result[row*size+col] = sum; 
    
}

__global__ void assignRowColKernel(float *result, minVal *minBlock,int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * size + col;

    if (row < size && col < size) {
        minBlock[idx].val = result[idx];   // Assign the value from the result matrix
        minBlock[idx].row = row;           // Assign the row index
        minBlock[idx].col = col;           // Assign the column index
    }
}



__global__ void reduceMinKernel(minVal *input, minVal *output, int gridSize,int size) {
   __shared__ minVal sharedMin[block * block];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int threadId = threadIdx.y * block + threadIdx.x;
    int idx = row * gridSize + col;

    if (row < gridSize && col < gridSize) {
        sharedMin[threadId] = input[idx];
    } else {
        sharedMin[threadId].val = FLT_MAX;  // Max float value to exclude invalid cells
    }

    __syncthreads();

    // Perform reduction within the block
    blockReduce(sharedMin, threadId);
    
    if (threadId == 0) {
        output[blockIdx.y * (gridSize / block) + blockIdx.x] = sharedMin[0];
    }
}

__device__ unsigned long long int PathAsULLI(path *path){
    unsigned long long int *ulli = reinterpret_cast<unsigned long long int*> (path) ;
    return *ulli; 

}

__device__ path* ULLIAsPath(unsigned long long int * ulli){
    path *var = reinterpret_cast<path*>(ulli); 

    return var; 
}

// __device__ __forceinline__ path *atomicMin(path *location, path *cpt){ // the forceinline directive would make sure that the function is inserted at the function call which would reduce the overhead. This doesn't guarentee that this would happen but if the overhead is reduced, the program overall would be faster. 
// // the function that we are going to use in this case to compare and find the min is atomic compare and swap. The function only takes int * or unsigned long long int as parameters 
// // as we are comparing two elements of the type path, we would have to reinterpret path as an ulli. 

//     unsigned long long int loc = PathAsULLI(location); 
//     while(cpt -> val < ULLIAsPath(&loc)-> val){ //ULLIAsPath(&loc)
//         unsigned long long int preCAS = loc; 
//         loc = atomicCAS((unsigned long long int*) location, preCAS, PathAsULLI(cpt));
//         if (loc == preCAS) break; 
//     }

//     return ULLIAsPath(&loc); 
// }

__device__ __forceinline__ path* atomicMin(path* eleAddress,path* newElement) 
{
    unsigned long long int newVal = *reinterpret_cast<unsigned long long int*>(newElement);
    unsigned long long int oldVal = *reinterpret_cast<unsigned long long int*>(eleAddress);
    while (newElement->val < reinterpret_cast<path*>(&oldVal)->val) {
        unsigned long long int returnedVal = atomicCAS(
            reinterpret_cast<unsigned long long int*>(eleAddress),oldVal,newVal);
        if (returnedVal == oldVal) break;
        oldVal = returnedVal;
    }
    return reinterpret_cast<path*>(&oldVal);//We use a custom function to convert the structure into an unsigned long long int and then apply atomicCAS function.
}

// Kernel 1 : computing costs 
__global__ void kernel1(minVal* d_initialMinBlock, int *vertex, int *edge, float *finalCost, path *costs, bool *mask , int size){
// host functions to compute the intermediate costs 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tid = row * size + col; 
    //if(d_initialMinBlock[tid].val < 3224){printf("not an ideal value for index %d,%d\n", row, col);} To check if the values at d_initialMinBlock are correct. 
    if(mask[tid]== true && row<size && col<size){
        mask[tid] = false; 
        for (int i = vertex[tid]; i<vertex[tid+1]; i++){
            path cpt; // cost + thread Ca+ Wa step in the pseudocode. 
            cpt.val = finalCost[tid] + d_initialMinBlock[tid].val;//edge[i] 
            cpt.index = tid; 
            //__syncthreads();
            atomicMin(&costs[edge[i]], &cpt); 
        }
    }
}

//Kernel 2 : Computing final costs 
__global__ void kernel2(bool *d_done, int *vertex, int *edge, float*finalCost, path *costs, bool *mask, int size){ // we don't need a mask for this one. 
// device functions to comp
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //printf("The value of d_done before being updated is %d\n",d_done); 
    int tid = row * size + col;
    if(finalCost[tid] > costs[tid].val && row < size && col < size){
        finalCost[tid] = costs[tid].val; // assign the min value to cost. 
        mask[tid] = true; 
        *d_done = false; 
    }
    //printf("The value of d_done after the if statement and being updated is %d\n",d_done); 

    costs[tid].val = finalCost[tid]; 

}

void findMin(minVal *A, minVal *B, minVal *C,int size){
    int gridSize = size; 
    dim3 blocksPerGrid1 (size/block, size/block); 

    reduceMinKernel<<<blocksPerGrid1, threadsPerBlock>>>(A, B, gridSize,size);
    cudaDeviceSynchronize();
    
    gridSize = (gridSize+block-1)/block; 
    blocksPerGrid1 = dim3((gridSize+block-1)/block, (gridSize+block-1)/block); // look into this grid/block.************************************ ATTENTION!!! CUZ THE GRIDSIZE HAS ALREADY BEEN DIVIDED BY BLOCK. MAYBE JUST KEEP IT AS GRIDsize/BLOCK
    while (gridSize > 1) {
        reduceMinKernel<<<blocksPerGrid1, threadsPerBlock>>>(B, B, gridSize, size);
        cudaDeviceSynchronize();
        gridSize = (gridSize+block-1)/block;
        int newblock = (gridSize+block-1)/block;
        blocksPerGrid1 = dim3((gridSize+block-1)/block, (gridSize+block-1)/block); 
    }

    CHECK(cudaMemcpy(C, B, (size / block) * (size / block)*sizeof(minVal), cudaMemcpyDeviceToHost));

}

void setup(int *vertex, int *edge, float *finalCost, path *costs, bool *mask , int size, minVal min1){// setup all the arrays so that we can perform the sssp steps. 
    int edgeIdx = 0; 
    // this block of code initializes the finalcost matrix, the cost matrix, (Ca and Cua r  espt), initializes the mask as we tlaked about in class, adn the edges array to get us the 
    //number of edges associated with each vertex and the the vertices array that store the information about edges at each vertex. 
    for (int i = 0; i < size; i ++) {
        for (int j =0; j<size; j++){
            mask[i*size +j] = false; 
            finalCost [i*size +j] = FLT_MAX; 
            costs[i*size +j].val = FLT_MAX; 
            costs[i*size +j].index = -1; 

            vertex[i*size +j] = edgeIdx; // This array would tell us where in the edges array would this point belong. So point 0 would start at 0, point 1 would start at 2 etc. 

            if ((j-1) >= 0) edge[edgeIdx++] = i*size+(j-1); //given this condition is true, th point has a edge on the left directed to the assigned point
            if ((i-1) >= 0) edge[edgeIdx++] = (i-1)*size +j; // edge on the top
            if ((j+1) < size) edge[edgeIdx++] =  i*size + (j+1); // edge on th right
            if ((i+1) < size) edge[edgeIdx++] = (i+1)*size + j; // edge on the bottom. 

        }  
    }
        //h_done[0] = false; 

        costs[min1.row*size + min1.col].val = 0.0f; // the cost of the source would be a zero as itwould be a t a distance zero from itself. 
        finalCost[min1.row*size + min1.col] = 0.0f; 
        mask[min1.row*size + min1.col] = true; // this is wher we start. 

} 

void showPath(path *s_path, int size, minVal min2, minVal min1){
    cv::Mat image = cv::Mat::zeros(size,size, CV_8UC3); 
    int next = min2.row * size + min2.col; // get the index for the destination.  
    bool last = false; 
    int count = 0; 

    

    while(last == false){ // check if it is the last element. If so then raise the flag so that we can stop. 
        count ++; // so we add 1 for each point
        if(s_path[next].index == -1) last = true; // when we find the last element set the flag. 
        if (last == true) next= min1.row * size + min1.col; // as this would be the last element the index would be -1 as it shouldn't be pointing to anything cuz it is the source. So we should update the -1 to the index of the source. 
        image.at<cv::Vec3b>((int)(next / size), (next % size)) = cv::Vec3b(0,165,255); // 0 , 165, 255 color for orange
        if (next == min2.row * size + min2.col) image.at<cv::Vec3b>((int)(next / size), (next % size)) = cv::Vec3b(255,255,255); // white for the destination. 
        if (next == min1.row * size + min1.col) image.at<cv::Vec3b>((int)(next / size), (next % size)) = cv::Vec3b(0,0,255); // blue for source. 
        if (s_path[next].val == 5813400.500000) printf("The index where the value is found is %d\n", s_path[next].index); 
        //if (s_path[next].val < minII.val) printf("Anomaly detected at index %d and the value is %f\n", s_path[next].index,s_path[next].val); 
        next = s_path[next].index; 
        
    }

    printf("Number of hops = %d\n", count-1); 

    cv::imshow("Shortest Path", image);
    while(true){
    int key = cv::waitKey(0); 
    if(key == 27) break; 
    }
}


void Mult(float *result, float *arr1, float *arr2, int n){
    float sum; 
    for(int i =0 ; i < n ; i++){
        for (int j = 0; j < n; j++){
            sum = 0.0; 
            for (int k =0 ; k< n; k++){
                sum+= arr1[i*n +k] * arr2[n*k +j]; 
            }
            result[i*n +j] = sum; 
        }
    }
    //printf("The min value is %f and the min index is (%d,%d)\n", minGlobal.val, minGlobal.index/ n, minGlobal.index % n);

}

int main(int argc, char* argv[]){

    int size = 4096; // default value of 4096
    if (argc < 2) {
        printf("Usage: %s <number>\n", argv[0]);
        return 1;
    }

    // Convert command-line argument to integer
    size = atoi(argv[1]);
    printf("The input size is %d\n",size); 
    struct timeval startTime; 
    struct timeval endTime;

    size_t size2 = size*size*sizeof(float); 
    size_t sizeStruct = (size/block)*(size/block)*sizeof(minVal); 

    float *test = (float*) malloc(size*size*sizeof(float)); 

    float *h_arr1 = (float*) malloc(size*size*sizeof(float)); 
    float *h_arr2 = (float*) malloc (size*size*sizeof(float)); 
    float *h_result = (float*) malloc (size*size*sizeof(float)); 
    minVal *h_minblock = (minVal*) malloc((size / block) * (size / block) * sizeof(minVal)); 

   

    minVal min1, min2; // elements of type struct to store the 2 minimums. 
    //minVal *hope = (minVal*) malloc ((size/block)* (size/block)* sizeof(minVal)); 

    int numTiles = size/ block; 

    float *d_arr1,*d_arr2,*d_result; 

    CHECK(cudaMalloc(&d_arr1, size2)); 
    CHECK(cudaMalloc(&d_arr2, size2)); 
    CHECK(cudaMalloc(&d_result, size2)); 

     for (int i = 0; i < size * size; i++) {
        h_arr1[i] = rand() / (float)1147654321;
        h_arr2[i] = rand() / (float)1147654321;
        h_result[i] = 0.0; 
    } // initialize matrices 

    //Mult(test, h_arr1, h_arr2, size); 

    dim3 blocksPerGrid( size/block , size/block);//(x,y,z) -> z is 1 by default.
    

    minVal *d_initialMinBlock; // to store the values of matrixMult and the associated indices. 
    CHECK(cudaMalloc(&d_initialMinBlock, size * size * sizeof(minVal)));
    minVal *h_initialMinBlock = (minVal*) malloc(size*size*sizeof(minVal)); 


    minVal *d_minBlock;
    CHECK(cudaMalloc(&d_minBlock, (size / block) * (size / block) * sizeof(minVal)));


    CHECK(cudaMemcpy(d_arr1, h_arr1, size2, cudaMemcpyHostToDevice)); 
    CHECK(cudaMemcpy(d_arr2, h_arr2, size2, cudaMemcpyHostToDevice)); 

    gettimeofday(&startTime, NULL);
    gettimeofday(&startMult, NULL); 
    matrixMult<<<blocksPerGrid, threadsPerBlock>>> (d_arr1, d_arr2, d_result, size); 
    cudaDeviceSynchronize();
    gettimeofday(&endMult, NULL);

    CHECK(cudaMemcpy(h_result, d_result, size2, cudaMemcpyDeviceToHost)); 

    

    assignRowColKernel<<<blocksPerGrid, threadsPerBlock>>>(d_result, d_initialMinBlock,size);
    cudaDeviceSynchronize();



    findMin(d_initialMinBlock, d_minBlock,h_minblock,size);
    printf("The minimum Value is %f at index %d, %d\n", h_minblock[0].val,h_minblock[0].row, h_minblock[0].col);

    min1 = h_minblock[0]; 
    // min1.row=69;
    // min1.col=66; 
    // min1.val= h_initialMinBlock[min1.row*size + min1.col].val;  
    //printf("val = %f\n", min1.val); 
    //printf("The value stored in min1 are  %.5f at index %d, %d\n", min1.val, min1.row, min1.col); 

    CHECK(cudaMemcpy(h_initialMinBlock, d_initialMinBlock, size*size*sizeof(minVal), cudaMemcpyDeviceToHost)); 
    h_initialMinBlock[((h_minblock[0].row * size) + h_minblock[0].col)].val = FLT_MAX; 

    
    CHECK(cudaMemcpy(d_initialMinBlock, h_initialMinBlock, size*size*sizeof(minVal), cudaMemcpyHostToDevice)); // Copy the value back to the device array that is going to be reduced. 
    findMin(d_initialMinBlock, d_minBlock,h_minblock,size);
    gettimeofday(&endMin, NULL);
    printf("The 2nd minimum Value is %f at index %d, %d\n", h_minblock[0].val,h_minblock[0].row, h_minblock[0].col);
    //minVal minII = h_minblock[0]; 
    min2 = h_minblock[0]; 
    // min2.row = 880;
    // min2.col = 919;  
    // min2.val= h_initialMinBlock[min1.row*size + min1.col].val; 
    //printf("val2= %f\n", min2.val) ; 
    //printf("The value stored in min1 are  %.5f at index %d, %d\n", min2.val, min2.row, min2.col); 
     //restoring the value of the first min in the output of the multiplication. 
    h_initialMinBlock[((min1.row * size) + min1.col)].val = min1.val;   
    CHECK(cudaMemcpy(d_initialMinBlock, h_initialMinBlock, size*size*sizeof(minVal), cudaMemcpyHostToDevice)); // copying to the device mem after restoring the old value of the min. 

    //printf("The value at the location of the first minimum is %.5f\n",h_initialMinBlock[((min1.row * size) + min1.col)].val ); 

    //printf("Value at index %d, %d, is %f for min1\n", min1.row, min1.col, h_initialMinBlock[min1.row*size+min1.col].val); 
    //printf("Value at index %d, %d, is %f for min1\n", min2.row, min2.col, h_initialMinBlock[min2.row*size+min2.col].val); 

    
    // gettimeofday(&endTime, NULL);

    // int host_int; 
    // int test_int; 

    // for (int i = 0; i< size; i++){
    //     for(int j=0; j<size; j++){
    //         //host_int = h_initialMinBlock[i*size+j].val * 1000; 
    //         //test_int = test[i*size+j] * 1000; 
    //         if(h_initialMinBlock[i*size+j].val < min2.val){
    //             printf("Incorrect value at index : %d,%d. and the value is %f\n", i,j, h_initialMinBlock[i*size+j].val); 
    //         }
    //     }
    // } //To test if matrix mUlt is correct. 
    printf("Multiplicatino done\n"); 


   // ******************************************************************************************************************************************************************************************************


    // Dijkstra's parallel implementation. 
    // h_initialMinBlock is the input to this part of the program. 
    // the cost of the path in this case is the sum of the edges only and not the sum of the edges as well as the vertices. 


    int numedge = (4 * 2 + (size-2)*3 * 4 + (((size-2)*(size-2))*4)); // 4 corners + other points on the extreme rows + every other point thats not in the extreme. 

    int *vertex = (int*) malloc((size * size + 1 ) * sizeof(int)); // size*size + 1 
    int *edge = (int*) malloc(numedge*sizeof(int)); 
    float *finalCost = (float*) malloc(size*size*sizeof(float)); // 
    path *costs= (path*) malloc(size*size*sizeof(path)); // this is the intermediate cost C_ua from the research paper.  
     // the edge array stores the index of the element that the edge is directed towards adn the vertex array stores the starting index of the of edges for each vertexin the edges array.
 
    bool *mask = (bool*) malloc(size*size*sizeof(bool)); // mask for the threads. 
    bool h_done = false; 
    minVal src, dst; 
    src.row = 69;
    src.col = 66;  
    dst.row = 880;
    dst.col = 919;  

    printf("Value at index %d, %d, is %f for min1\n", src.row, src.col, h_initialMinBlock[src.row*size+src.col].val); 
    printf("Value at index %d, %d, is %f for min1\n", dst.row, dst.col, h_initialMinBlock[dst.row*size+dst.col].val); 


    int* d_vertex; 
    int *d_edge; 
    float* d_finalCost; 
    path *d_costs;
    bool *d_mask; 
    bool *d_done; // flag to indicate the that device is done.  copied to the host flag. 

    CHECK(cudaMalloc(&d_vertex, ((size * size + 1) * sizeof(int)))); // size*size + 1
    CHECK(cudaMalloc(&d_edge, numedge*sizeof(int))); 
    CHECK(cudaMalloc(&d_finalCost, size*size*sizeof(float))); 
    CHECK(cudaMalloc(&d_costs, size*size*sizeof(path)));
    CHECK(cudaMalloc(&d_mask,size*size*sizeof(bool))); // memory allocated for all these required devices in the gpu. 
    CHECK(cudaMalloc(&d_done, sizeof(bool))); 
    // bool h_done = (bool*) malloc (sizeof(bool)); 
    // h_done[0] = false; //(bool*) malloc (sizeof(bool)); 
    gettimeofday(&ssspStart, NULL);
    setup(vertex,edge, finalCost, costs, mask, size, src); 
    vertex[size*size] = numedge; // May or may not use this. 

    //printf("The cost at the source is %f", costs[min1.row*size+1+min1.col].val); 

    
    
    CHECK(cudaMemcpy(d_vertex, vertex,((size * size + 1 ) * sizeof(int)),cudaMemcpyHostToDevice)); // size*size + 1
    CHECK(cudaMemcpy(d_edge, edge,numedge*sizeof(int) ,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_finalCost, finalCost,size*size*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_costs, costs, size*size*sizeof(path), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mask, mask,size*size*sizeof(bool),cudaMemcpyHostToDevice)); // copying the memory from the host to the device. 
    //CHECK(cudaMemcpy(d_done, h_done, sizeof(bool), cudaMemcpyHostToDevice)); 

    dim3 blocksPerGrid2(size/block,size/block); 
    dim3 threadsPerBlock2(block,block); 

    while(h_done == false){

        //printf("Entered the while loop.\n"); 

        h_done = true; 

        CHECK(cudaMemcpy(d_done, &h_done, sizeof(bool), cudaMemcpyHostToDevice)); 
        //kernel1(minVal* d_initialMinBlock, int *vertex, int *edge, float *finalCost, path *costs, bool *mask , int size){

        //call to kernel 1 to compute the intermediate costs 
        kernel1<<<blocksPerGrid2, threadsPerBlock2>>>(d_initialMinBlock,d_vertex,d_edge,d_finalCost,d_costs,d_mask,size); 
        cudaDeviceSynchronize(); 

        //call to kernel 2 to compute the final costs. 
        //(bool *d_done, int *vertex, int *edge, float*finalCost, path *costs, bool *mask, int size
        kernel2<<<blocksPerGrid2,threadsPerBlock2>>>(d_done, d_vertex,d_edge,d_finalCost, d_costs, d_mask, size); 
        cudaDeviceSynchronize(); 

        //printf("after kernel 2.\n"); 

        CHECK(cudaMemcpy(&h_done,d_done,sizeof(bool), cudaMemcpyDeviceToHost)); // copy the value of the device done pointer that you get from kernel 2. 
        
        //printf("after copying the done pointer.\n"); 

    }


    CHECK(cudaMemcpy(costs, d_costs, size*size*sizeof(path), cudaMemcpyDeviceToHost)); 

    printf("The weight of the path to get to the second min value from the first min value is %f.\n", costs[dst.row * size + dst.col].val);
     //float actualWeight = costs[min2.row * size + min2.col].val + h_initialMinBlock[min1.row * size + min1.col].val - h_initialMinBlock[min2.row * size + min2.col].val ; 
     //printf("Actual weight = %f", actualWeight); 

     gettimeofday(&endTime, NULL);
     for (int i = 0; i< size; i++){
        for(int j=0; j<size; j++){
            //host_int = h_initialMinBlock[i*size+j].val * 1000; 
            //test_int = test[i*size+j] * 1000; 
            if(costs[i*size+j].val < min2.val){
                printf("Incorrect value at index : %d,%d. and the value is %f\n", i,j, costs[i*size+j].val); 
            }
        }
    }

    showPath(costs, size, dst, src); 

    long seconds, uSeconds, secondsMult, secondsMin, uMult, uMin, secondSSSP, uSSSP;
    seconds = endTime.tv_sec - startTime.tv_sec;
    uSeconds = endTime.tv_usec - startTime.tv_usec;
    secondsMult = endMult.tv_sec - startMult.tv_sec; 
    uMult = endMult.tv_usec - startMult.tv_usec; 
    secondsMin = endMin.tv_sec - endMult.tv_sec; 
    uMin = endMin.tv_usec - endMult.tv_usec; 
    secondSSSP = endTime.tv_sec - ssspStart.tv_sec; 
    uSSSP = endTime.tv_usec - ssspStart.tv_usec; 
    double elapsed = seconds + uSeconds / 1000000.0;
    double elapsedMult = secondsMult + uMult/1000000.0;
    double elapsedMin = secondsMin + uMin/1000000.0;
    double elapsedSSSP = secondSSSP + uSSSP/1000000.0;
    cout <<"Elapsed Time : " << elapsed << endl; 
    cout <<"Multiplication Time: "<< elapsedMult <<endl; 
    cout <<"Min time: "<<elapsedMin<<endl;
    cout <<"SSSP Time: "<< elapsedSSSP << endl; 

    free(h_arr1); 
    free(h_arr2); 
    free(h_result); 
    free(h_minblock); 
    free(h_initialMinBlock); 
    (cudaFree(d_arr1)); 
    (cudaFree(d_arr2)); 
    (cudaFree(d_result)); 
    (cudaFree(d_minBlock)); 
    (cudaFree(d_initialMinBlock)); 



    return 0; 
}