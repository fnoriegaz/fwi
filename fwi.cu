///////////////////////////////////////////////////////////////////////////////////////////////////
#include<iostream>
#include<vector>
#include<map>
#include<unordered_map>


#define CUDA_CHECK(call) {                                 \
    cudaError_t err = call;                               \
    if (err != cudaSuccess) {                            \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " in " << __FILE__ << " at line " << __LINE__ \
                  << std::endl;                           \
        exit(EXIT_FAILURE);                               \
    }                                                    \
}


//use ricker wavelet to create source
void create_source(float *source, int timesamples, float dt, float freq){

    float wp = 2.0*M_PI*freq;
    for(int i=0; i<timesamples; i++){
        float tau = i*dt;
        source[i] = (1 - 0.5*wp*wp*tau*tau)*exp(-0.25*wp*wp*tau*tau);
    }
}


__global__
void compute_p(float *p, float *vx, float *vy, float *c, float *rho,
               float dx, float dy, float dt, int cols, int rows){

    //__shared__ float shared_p[16*3][16];
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;
    int tid = idx + cols * idy;

    //compute vx
    if(idx > 0 && idx < cols-1 && idy > 0 && idy < rows-1){
        vx[tid] = vx[tid] + (dt/rho[tid]) * (-(1./2)*p[tid-1] + (1./2)*p[tid+1])/dx;
        vy[tid] = vy[tid] + (dt/rho[tid]) * (-(1./2)*p[tid-1*cols] + (1./2)*p[tid+1*cols])/dy;
    }
     __syncthreads();
    if(idx > 0 && idx < cols-1 && idy > 0 && idy < rows-1){
        p[tid] = p[tid] - (dt * rho[tid]*c[tid]*c[tid]) * (
            -((1./2)*vx[tid-1] + (1./2)*vx[tid+1])/dx
            -((1./2)*vy[tid-1*cols] + (1./2)*vy[tid+1*cols])/dy
        );
    }

}


class PressureField{
private:
    int rows, cols, time_samples;
    float dx, dy, dt;
    float *p = nullptr;
    float *rho = nullptr;
    float *c = nullptr;
    float *vx = nullptr;
    float *vy = nullptr;
    float *source = nullptr;

public:
    PressureField(float *c, float *rho, int rows, int cols, float dx, float dy, float dt, int time_samples);
    void launch_compute_p();
    ~PressureField();

};

PressureField::PressureField(float *c, float *rho, int rows, int cols, float dx, float dy, float dt, 
                            int time_samples){
    
    this->rows = rows;
    this->cols = cols;
    this->time_samples = time_samples;
    this->c = c;
    this->rho = rho;
    cudaMallocManaged(&this->p, cols*rows*sizeof(float));
    cudaMallocManaged(&this->vx, cols*rows*sizeof(float));
    cudaMallocManaged(&this->vy, cols*rows*sizeof(float));
    cudaMallocManaged(&this->source, time_samples*sizeof(float));

    cudaMemset(&p, 0.0, cols*rows*sizeof(float));
    cudaMemset(&vx, 0.0, cols*rows*sizeof(float));
    cudaMemset(&vy, 0.0, cols*rows*sizeof(float));
    cudaMemset(&source, 0.0, time_samples*sizeof(float));
    
}

void PressureField::launch_compute_p(){
    int time_samples = 3500;
    dim3 block_size(16,16,1);
    dim3 grid_size(512/16,512/16,1);

    for(int s=0;s<100;s++){
        for(int t=0;t<time_samples;t++){
            compute_p<<<grid_size,block_size>>>(p, vx, vy, c, rho, dx, dy, dt, cols, rows);
            CUDA_CHECK(cudaDeviceSynchronize());
            this->p[512/2 + cols*512/2] += this->source[t];
        }
    }
    
    
}

PressureField::~PressureField(){
    cudaFree(p);
    cudaFree(vx);
    cudaFree(vy);
    cudaFree(source);
}



int main(){
    
    int rows = 512;
    int cols = 512;
    int time_samples = 3500;
    float dx = 25.;
    float dy = 25.;
    float dt = 0.001;
    float freq = 3.;
    float *source = nullptr;
    float *c = nullptr;
    float *rho = nullptr;

    cudaMallocManaged(&source, time_samples*sizeof(float));
    cudaMallocManaged(&c, rows*cols*sizeof(float));
    cudaMallocManaged(&rho, rows*cols*sizeof(float));

    
    for(int i=0;i<rows*cols; i++){
        c[i] = 2500.0;
        rho[i] = 997.0;
    }
    create_source(source,time_samples,dt,freq);
    PressureField pressure = PressureField(c,rho,rows,cols,dx,dy,dt,time_samples);
    
    pressure.launch_compute_p();


    cudaFree(c);
    cudaFree(rho);
    
    return 0;
}