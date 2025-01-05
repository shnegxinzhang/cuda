/*-----------------------------------------------
 * 请在此处填写你的个人信息
 * 学号:SC24219042
 * 姓名:张胜欣
 * 邮箱:zhang2048525561@163.com
 ------------------------------------------------*/

#include <cuda_runtime.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;

//全局变量，交换指针
char * tmp;
#define AT(x, y, z) grid[(x) * N * N + (y) * N + z]
#define EXCHANGE_PTR(x,y) {tmp=x; x=y; y=tmp;}

//using std::cin, std::cout, std::endl;
//using std::ifstream, std::ofstream;

// 存活细胞数
int population(int N, char *grid)
{
    int result = 0;
    for (int i = 0; i < N * N * N; i++)
        result += grid[i];
    return result;
}

// 打印世界状态
void print_grid(int N, char *grid)
{
    // 仅在N较小(<= 32)时用于Debug
    if (N > 32)
        return;
    for (int x = 0; x < N; x++)
    {
        for (int y = 0; y < N; y++)
        {
            for (int z = 0; z < N; z++)
            {
                if (AT(x, y, z))
                    cout << "O ";
                else
                    cout << "* ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << "population: " << population(N, grid) << endl;
}

// CUDA 核函数，执行 3D 生命游戏的更新规则
__global__ void life3d_kernel(int N, char* grid, char* nextGrid) {
// 计算当前线程在三维网格中的位置
    int x = (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (blockIdx.y * blockDim.y + threadIdx.y);
    int z = (blockIdx.z * blockDim.z + threadIdx.z);
// 若超出网格范围则不处理
    if (x >= N || y >= N || z >= N)
        return;

    int dx,dy,dz;
    int nx,ny,nz,idx;
    // 初始化存储存活邻居的数量alive_nei_num
    int alive_nei_num = 0;
    // 遍历周围 27 个邻居（包括自身，但自身会被跳过）
    for (dx = -1; dx <= 1; dx++) {
        for (dy = -1; dy <= 1; dy++) {
            for (dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0)
                    continue;
                // 考虑边界循环，使用取模操作
                nx = (x + dx + N) % N;
                ny = (y + dy + N) % N;
                nz = (z + dz + N) % N;
                // 统计存活邻居的数量
                alive_nei_num += AT(nx, ny, nz);
            }
        }
    }
    // 计算当前元素在一维数组中的索引
    idx = x * N * N + y * N + z;
    // 应用生命游戏规则更新细胞状态
    if (AT(x, y, z) && (alive_nei_num < 5 || alive_nei_num > 7))
        nextGrid[idx] = 0;
    else if (!AT(x, y, z) && alive_nei_num == 6)
        nextGrid[idx] = 1;
    else
        nextGrid[idx] = AT(x, y, z);
}

// 主函数，在 GPU 上运行 3D 生命游戏
void life3d_gpu(int N, char* grid, int T, int threadBlockSize) {
// 分配 GPU 内存
//定义需要的局部变量
    char *grid_c, *nextGrid;
    cudaMalloc(&grid_c, N * N * N);
    cudaMalloc(&nextGrid, N * N * N);
    // 将数据从主机复制到 GPU 内存
    cudaMemcpy(grid_c, grid, N * N * N, cudaMemcpyHostToDevice);
    // 定义线程块和线程块网格的大小
    dim3 threadsPerBlock(threadBlockSize, threadBlockSize, threadBlockSize); //thread x,y,z
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (N + threadsPerBlock.z - 1) / threadsPerBlock.z);
    // 迭代 T 次更新状态
    for (int t = 0; t < T; t++) {
        life3d_kernel<<<numBlocks, threadsPerBlock>>>(N, grid_c, nextGrid);
        cudaDeviceSynchronize();
    EXCHANGE_PTR(grid_c,nextGrid);
    }
    // 将最终结果从 GPU 复制回主机
    cudaMemcpy(grid, grid_c, N * N * N, cudaMemcpyDeviceToHost);

    cudaFree(grid_c);
    cudaFree(nextGrid);
}

// 读取输入文件
// input_file 输入文件的路径
// buffer 存储读取数据的缓冲区
void read_file(char *input_file, char *buffer)
{
    ifstream file(input_file, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        cout << "Error: Could not open file " << input_file << std::endl;
        exit(1);
    }
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer, file_size))
    {
        std::cerr << "Error: Could not read file " << input_file << std::endl;
        exit(1);
    }
    file.close();
}

// 写入输出文件
// output_file 输出文件的路径
// buffer 存储要写入数据的缓冲区
void write_file(char *output_file, char *buffer, int N)
{
    ofstream file(output_file, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        cout << "Error: Could not open file " << output_file << std::endl;
        exit(1);
    }
    file.write(buffer, N * N * N);
    file.close();
}

int main(int argc, char **argv)
{
    // cmd args
    if (argc < 5)
    {
        cout << "usage: ./life3d N T input output" << endl;
        return 1;
    }
//定义需要的变量，包括分块尺寸等
    int threadBlockSize=4;					//块尺寸
    int N = std::stoi(argv[1]);					//世界网格大小
    int T = std::stoi(argv[2]);					//迭代次数
    char *input_file = argv[3];				//初始文件
    char *output_file = argv[4];				//输出文件
    char *grid = (char *)malloc(N * N * N);		// 分配主机内存存储网格数据
    read_file(input_file, grid);					// 从文件读取初始状态
    int start_pop = population(N, grid);			// 计算初始存活细胞的数量
    auto start_time = std::chrono::high_resolution_clock::now();		//统计时间end_time-start_time
    // 在 GPU 上运行 3D 生命游戏
    life3d_gpu(N, grid, T, threadBlockSize);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    // 计算最终存活细胞的数量
    int final_pop = population(N, grid);
    //将计算结果存入文件中
    write_file(output_file, grid, N);
    
    cout << "TPB:" << threadBlockSize << "*" << threadBlockSize << "*" << threadBlockSize << endl;
    cout << "GPU:" << endl;
    cout << "start population: " << start_pop << endl;
    cout << "final population: " << final_pop << endl;
    double time = duration.count();
    cout << "time: " << time << "s" << endl;
    cout << "cell per sec: " << T / time * N * N * N << endl;
    
    free(grid);
    return 0;
}

