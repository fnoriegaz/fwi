///////////////////////////////////////////////////////////////////////////////////////////////////
#include<iostream>
#include<vector>
#include<map>
#include<unordered_map>



__global__
void compute_p(float *p, float *rho, float *p, float *rho, float *p, float *rho){

    __shared__ float shared_p[16*3][16];
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;
    int tid = idx + cols * tidy;

    //compute vx
    if(idx > 0 && idx < cols && idy > 0 && idy < rows){
        vx[tid] = vx[tid] + (dt/rho[tid]) * (-(1./2)*p[tid-1] + (1./2)*p[tid+1]);
        vy[tid] = vy[tid] + (dt/rho[tid]) * (-(1./2)*p[tid-1*cols] + (1./2)*p[tid+1*cols]);
    }

    if(idx > 0 && idx < cols && idy > 0 && idy < rows){
        p[tid] = p[tid] - (dt * rho[tid]*c[tid]*c[tid]) * (
            -(1./2)*vx[tid-1] + (1./2)*vx[tid+1]
            -(1./2)*p[tid-1*cols] + (1./2)*p[tid+1*cols]
        );
    }

}


class PressureField{
private:
    int rows, cols;
    float dx, dy, dt;
    float *p;
    float *rho;
    float *c;
    float *vx;
    float *vy;

public:
    PressureField(const *float c, const *float rho,
                 const int rows, const int cols,
                 float dx, float dy, float dt);
    ~PressureField();

    __global__ void compute_p();
};

PressureField::PressureField(const *float c, const *float rho,
                            const int rows, const int cols, 
                            float dx, float dy, float dt){
    
    this->rows = rows;
    this->cols = cols;
    memcpy(this->c, c, cols*rows*sizeof(float));
    memcpy(this->rho, rho, cols*rows*sizeof(float));
    
}

__global__
void PressureField::launch_compute_p(){
    dim3 block_size(16,16,1);
    dim3 grid_size(512/16,512/16,1);
    
}

PressureField::~PressureField(){
}



int main(){
    
    int rows = 512;
    int cols = 512;
    float dx = dy = 25.;
    float dt = 0.001;
    float *c = nullptr;
    float *rho = nullptr;

    cudaMallocManaged(&c, rows*cols*sizeof(float));
    cudaMallocManaged(&rho, rows*cols*sizeof(float));

    PressureField pressure = PressureField(c, rho, rows, cols, );
    for(int i=0;i<rows*cols; i++){
        c[i] = 2500.0;
        rho[i] = 997.0;
    }
    
    return 0;
}