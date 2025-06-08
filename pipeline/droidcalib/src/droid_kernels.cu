#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <math.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

// #include "utils.cuh"

#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;
typedef std::vector<std::vector<long>> graph_t;
typedef std::vector<torch::Tensor> tensor_list_t;



#define MIN_DEPTH 0.25

#define THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + THREADS - 1) / THREADS)


#define GPU_1D_KERNEL_LOOP(k, n) \
  for (size_t k = threadIdx.x; k<n; k += blockDim.x)


__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid +  8];
  sdata[tid] += sdata[tid +  4];
  sdata[tid] += sdata[tid +  2];
  sdata[tid] += sdata[tid +  1];
}

__device__ void blockReduce(volatile float *sdata) {
  unsigned int tid = threadIdx.x;
  __syncthreads();

  // if (threadIdx.x < 256) {sdata[tid] += sdata[tid + 256]; } __syncthreads();
  if (threadIdx.x < 128) {sdata[tid] += sdata[tid + 128]; } __syncthreads();
  if (threadIdx.x <  64) {sdata[tid] += sdata[tid +  64]; } __syncthreads();

  if (tid < 32) warpReduce(sdata, tid);
  __syncthreads();
}


__device__ void
actSO3(const float *q, const float *X, float *Y) {
  float uv[3];
  uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

__device__  void
actSE3(const float *t, const float *q, const float *X, float *Y) {
  actSO3(q, X, Y);
  Y[3] = X[3];
  Y[0] += X[3] * t[0];
  Y[1] += X[3] * t[1];
  Y[2] += X[3] * t[2];
}

__device__ void
adjSE3(const float *t, const float *q, const float *X, float *Y) {
  float qinv[4] = {-q[0], -q[1], -q[2], q[3]};
  actSO3(qinv, &X[0], &Y[0]);
  actSO3(qinv, &X[3], &Y[3]);

  float u[3], v[3];
  u[0] = t[2]*X[1] - t[1]*X[2];
  u[1] = t[0]*X[2] - t[2]*X[0];
  u[2] = t[1]*X[0] - t[0]*X[1];

  actSO3(qinv, u, v);
  Y[3] += v[0];
  Y[4] += v[1];
  Y[5] += v[2];
}

__device__ void 
relSE3(const float *ti, const float *qi, const float *tj, const float *qj, float *tij, float *qij) {
  qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1],
  qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2],
  qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0],
  qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2],

  actSO3(qij, ti, tij);
  tij[0] = tj[0] - tij[0];
  tij[1] = tj[1] - tij[1];
  tij[2] = tj[2] - tij[2];
}

__device__ void
rotmat(const float *qij, float R[][3]){

  // storage of quaternion as x, y, z, w
  float s = 1.0; // unit quaternions

  R[0][0] = 1 - 2 * s * (qij[1]*qij[1] + qij[2]*qij[2]);
  R[0][1] = 2 * s * (qij[0]*qij[1] - qij[2]*qij[3]);
  R[0][2] = 2 * s * (qij[0]*qij[2] + qij[1]*qij[3]);

  R[1][0] = 2 * s * (qij[0]*qij[1] + qij[2]*qij[3]);
  R[1][1] = 1 - 2 * s * (qij[0]*qij[0] + qij[2]*qij[2]);
  R[1][2] = 2 * s * (qij[1]*qij[2] - qij[0]*qij[3]);

  R[2][0] = 2 * s * (qij[0]*qij[2] - qij[1]*qij[3]);
  R[2][1] = 2 * s * (qij[1]*qij[2] + qij[0]*qij[3]);
  R[2][2] = 1 - 2 * s * (qij[0]*qij[0] + qij[1]*qij[1]);

  
}

  
__device__ void
expSO3(const float *phi, float* q) {
  // SO3 exponential map
  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta_p4 = theta_sq * theta_sq;

  float theta = sqrtf(theta_sq);
  float imag, real;

  if (theta_sq < 1e-8) {
    imag = 0.5 - (1.0/48.0)*theta_sq + (1.0/3840.0)*theta_p4;
    real = 1.0 - (1.0/ 8.0)*theta_sq + (1.0/ 384.0)*theta_p4;
  } else {
    imag = sinf(0.5 * theta) / theta;
    real = cosf(0.5 * theta);
  }

  q[0] = imag * phi[0];
  q[1] = imag * phi[1];
  q[2] = imag * phi[2];
  q[3] = real;

}

__device__ void
crossInplace(const float* a, float *b) {
  float x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}

__device__ void
expSE3(const float *xi, float* t, float* q) {
  // SE3 exponential map

  expSO3(xi + 3, q);
  float tau[3] = {xi[0], xi[1], xi[2]};
  float phi[3] = {xi[3], xi[4], xi[5]};

  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta = sqrtf(theta_sq);

  t[0] = tau[0]; 
  t[1] = tau[1]; 
  t[2] = tau[2];

  if (theta > 1e-4) {
    float a = (1 - cosf(theta)) / theta_sq;
    crossInplace(phi, tau);
    t[0] += a * tau[0];
    t[1] += a * tau[1];
    t[2] += a * tau[2];

    float b = (theta - sinf(theta)) / (theta * theta_sq);
    crossInplace(phi, tau);
    t[0] += b * tau[0];
    t[1] += b * tau[1];
    t[2] += b * tau[2];
  }
}



__device__ void
proj_transform_mei(const float* uvi, const float* qij, const float* tij, 
        const float fx, const float fy, const float cx, const float cy, 
        const float xi, const float disp, float* uvj) {

  const float xs = (uvi[0] - cx) / fx;
  const float ys = (uvi[1] - cy) / fy;
  const float beta = (xi + sqrt( 1 + ( 1 - xi * xi) * (xs * xs  + ys * ys))) 
                     / (1 + xs * xs  + ys * ys);

  float Xi[4];
  float Xj[4];

  Xi[0] = beta * xs / (beta - xi);
  Xi[1] = beta * ys / (beta - xi);
  Xi[2] = 1.0;
  Xi[3] = disp;

  actSE3(tij, qij, Xi, Xj);

  const float d = (Xj[2] < MIN_DEPTH) ? 0.0 : 1.0 / Xj[2];
  const float z = (Xj[2] < MIN_DEPTH) ? 0.0 : Xj[2];
  const float r = sqrt(Xj[0] * Xj[0] + Xj[1] * Xj[1] + z * z);
  const float mei_factor = ((z + xi * r) < MIN_DEPTH) ? 0.0 : 1.0 / (z + xi * r);

  uvj[0] = fx * Xj[0] * mei_factor + cx ;
  uvj[1] = fy * Xj[1] * mei_factor + cy ;
}



__device__ void finite_difference_mei(const float fx, const float fy, const float cx, const float cy, 
                                       const float xi, const float* uvi, const float* qij, 
                                       const float* tij, const float disp, const float mu,
                                       const int direction, const int component, float* Jc) {
    float u_low, u_up;
    float delta = (direction == 4) ? 0.01 * mu : mu; // for xi we use a smaller step size
    float uvj[2];

    float offset[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    offset[direction] = delta;

    proj_transform_mei(uvi, qij, tij, fx - offset[0], fy - offset[1], cx - offset[2], 
                      cy - offset[3], xi - offset[4], disp, uvj);
    u_low = uvj[component];
    
    proj_transform_mei(uvi, qij, tij, fx + offset[0], fy + offset[1], cx + offset[2], 
                       cy + offset[3], xi + offset[4], disp, uvj);
    u_up = uvj[component];

    Jc[direction] = (u_up - u_low) / (2 * delta);
}


__global__ void projective_transform_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> target,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> vs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eii,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eij,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Cii,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> bz,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Calib,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> CalibPose,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> CalibDepth,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> qs,
    const bool opt_intr,
    const int n_intr,
    const int model_id  // 0: pinhole, 1: mei, 2: focal
    )
{
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  int ix = static_cast<int>(ii[block_id]);
  int jx = static_cast<int>(jj[block_id]);

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;
  __shared__ float xi;


  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4]; 
  __shared__ float R[3][3]; 

  if (thread_id == 0) {
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];

    if (model_id == 1){
      xi = intrinsics[4];
    }

  }

  __syncthreads();

  // stereo frames
  if (ix == jx) {
    if (thread_id == 0) {
      tij[0] =  -0.1;
      tij[1] =     0;
      tij[2] =     0;
      qij[0] =     0;
      qij[1] =     0;
      qij[2] =     0;
      qij[3] =     1;
    }
  }

  else {

    if (thread_id < 3) {
      ti[thread_id] = poses[ix][thread_id];
      tj[thread_id] = poses[jx][thread_id];
    }

    if (thread_id < 4) {
      qi[thread_id] = poses[ix][thread_id+3];
      qj[thread_id] = poses[jx][thread_id+3];
    }

    __syncthreads();

    if (thread_id == 0) {
      relSE3(ti, qi, tj, qj, tij, qij);
      rotmat(qij, R);
    }

  }

  __syncthreads();

  //points 
  float Xi[4];
  float Xj[4];

  // jacobians
  float Jx[12];
  float Jz;
  float Jc[5]; //n_intr=4 or 5; using 5 to have sufficient space

  float* Ji = &Jx[0];
  float* Jj = &Jx[6];

  // pose hessian
  float hij[12*(12+1)/2];

  // calib hessian
  float cij[5*(5+1)/2]; 
  
  // calib-pose-blcok 
  float cpij[2][5][6]; 
  float q[5]; 

  float vi[6], vj[6];

  // initialize everything with zero
  int l;
  for (l=0; l<12*(12+1)/2; l++) {
    hij[l] = 0;
  }

  for (l=0; l<2; l++) {
    for (int n=0; n<n_intr; n++) {
      for (int m=0; m<6; m++) {
        cpij[l][n][m] = 0;
      }
    }  
  }

  for (l=0; l<n_intr*(n_intr+1)/2; l++) {
    cij[l] = 0;
  }

  for (int n=0; n<6; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }
  
  for (int n=0; n<n_intr; n++) {
    q[n] = 0;
  }

  __syncthreads();

  GPU_1D_KERNEL_LOOP(k, ht*wd) {

    const int i = k / wd;
    const int j = k % wd;

    const float u = static_cast<float>(j);
    const float v = static_cast<float>(i);

    float uvi[2];
    float uvj[2];
    uvi[0] = u;
    uvi[1] = v;

    float ru;
    float rv;

    if (model_id == 0 || model_id == 2){
      Xi[0] = (u - cx) / fx;
      Xi[1] = (v - cy) / fy;
      Xi[2] = 1;
      Xi[3] = disps[ix][i][j];
    }
    else {
      const float xs = (u - cx) / fx;
      const float ys = (v - cy) / fy;
      const float beta = (xi + sqrt( 1 + ( 1 - xi * xi) * (xs * xs  + ys * ys))) 
                         / (1 + xs * xs  + ys * ys);

      Xi[0] = beta * xs / (beta - xi);
      Xi[1] = beta * ys / (beta - xi);
      Xi[2] = 1;
      Xi[3] = disps[ix][i][j];
    }
    // transform homogenous point
    actSE3(tij, qij, Xi, Xj);

    const float x = Xj[0];
    const float y = Xj[1];
    const float h = Xj[3];

    const float d = (Xj[2] < MIN_DEPTH) ? 0.0 : 1.0 / Xj[2];
    const float d2 = d * d;
    const float z = (Xj[2] < MIN_DEPTH) ? 0.0 : Xj[2];
    const float r = sqrt(x * x + y * y + z * z);
    

    float wu = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][0][i][j];
    float wv = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][1][i][j];

    if (model_id == 0 || model_id == 2){
      ru = target[block_id][0][i][j] - (fx * d * x + cx);
      rv = target[block_id][1][i][j] - (fy * d * y + cy);
    }
    else {
      const float mei_factor = ((z + xi * r) < MIN_DEPTH) ? 0.0 : 1.0 / (z + xi * r);
      ru = target[block_id][0][i][j] - (fx * x * mei_factor + cx);
      rv = target[block_id][1][i][j] - (fy * y * mei_factor + cy);
    }

    // x - coordinate
    if (model_id == 0 || model_id == 2){
      Jj[0] = fx * (h*d);
      Jj[1] = fx * 0;
      Jj[2] = fx * (-x*h*d2);
      Jj[3] = fx * (-x*y*d2);
      Jj[4] = fx * (1 + x*x*d2);
      Jj[5] = fx * (-y*d);
      Jz = fx * (tij[0] * d - tij[2] * (x * d2));
    }
    else {
      const float mf = ((z + xi * r) < MIN_DEPTH) ? 0.0 : 1.0 / (z + xi * r);
      const float mf2 = mf * mf;
      const float a1 = fx * mf;
      const float a2 = fx * xi * x * x / r * mf2;
      const float a3 = fx * xi * x * y / r * mf2;
      const float a4 = fx * x * (xi * z / r + 1.0) * mf2;

      Jj[0] = h * (a1 - a2);
      Jj[1] = -h * a3;
      Jj[2] = -h * a4;

      Jj[3] = z * a3 - y * a4;
      Jj[4] = z * (a1 - a2) + x * a4;
      Jj[5] = - y * (a1 - a2)  - x * a3;

      Jz = tij[0] * (a1 - a2) - tij[1] * a3 
           - tij[2] * a4;

    }

    Cii[block_id][k] = wu * Jz * Jz;
    bz[block_id][k] = wu * ru * Jz;

    if (ix == jx) wu = 0;


    adjSE3(tij, qij, Jj, Ji);
    for (int n=0; n<6; n++) Ji[n] *= -1;

    l=0;
    for (int n=0; n<12; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += wu * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<6; n++) {
      vi[n] += wu * ru * Ji[n];
      vj[n] += wu * ru * Jj[n];

      Eii[block_id][n][k] = wu * Jz * Ji[n];
      Eij[block_id][n][k] = wu * Jz * Jj[n];
    }

    if (opt_intr){

      // calib jacobian pinhole
      if (model_id == 0){
        Jc[0] = -R[0][0] * (u-cx) * (1.0/fx) * d 
                + R[2][0] * (u-cx) * (1.0/fx) * x * d2 + x * d;
        Jc[1] = -R[0][1] * (v-cy) * (1.0/fy) * (1.0/fy) * fx * d 
                + R[2][1] * (v-cy) * (1.0/fy) * (1.0/fy) * fx * x * d2;
        Jc[2] = -R[0][0] * d + x * d2 * R[2][0] + 1.0;
        Jc[3] = -R[0][1] * fx * (1.0/fy) * d + fx * x * d2 * (1.0/fy) * R[2][1];
      }
      // calib jacobian focal model
      else if (model_id == 2){
        float zi = 1;
        float a1 =  (u-cx) * zi * (1.0/fx) * (1.0/fx);
        float a2 =  (v-cy) * zi * (1.0/fy) * (1.0/fy);
        float l1 = - R[0][0] * a1 - R[0][1] * a2;
        float l2 = - R[1][0] * a1 - R[1][1] * a2;
        float l3 = - R[2][0] * a1 - R[2][1] * a2;
        Jc[0] = fx * d * l1 - fx * x * d2 * l3 + x * d;
      }
      else{ 
        // calib jacobian mei model: numerical approximation because analytical 
        // expressions are pages long
        const float mu = 0.01;
        const float disp = disps[ix][i][j];
        const int component = 0;

        for (int direction = 0; direction < 5; direction++) {
            finite_difference_mei(fx, fy, cx, cy, xi, uvi, qij, tij, disp, 
                                  mu, direction, component, Jc);
        }
      }
      
      // calib block
      l=0;
      for (int n=0; n<n_intr; n++) {
        for (int m=0; m<=n; m++) {
          cij[l] += wu * Jc[n] * Jc[m];
          l++;
        }
      }

      // calib-pose block
      for (int n=0; n<n_intr; n++) {
        for (int m=0; m<6; m++) {
          cpij[0][n][m] += wu * Jc[n] * Ji[m];
          cpij[1][n][m] += wu * Jc[n] * Jj[m];
        }
      }


      for (int n=0; n<n_intr; n++) {
        q[n] += wu * ru * Jc[n];
        CalibDepth[block_id][k][n] = wu * Jz * Jc[n];
      }

    }

    // y - coordinate
    if (model_id == 0 || model_id == 2){
      Jj[0] = fy * 0;
      Jj[1] = fy * (h*d);
      Jj[2] = fy * (-y*h*d2);
      Jj[3] = fy * (-1 - y*y*d2);
      Jj[4] = fy * (x*y*d2);
      Jj[5] = fy * (x*d);
      Jz = fy * (tij[1] * d - tij[2] * (y * d2));
    }
    else{
      const float mf = ((z + xi * r) < MIN_DEPTH) ? 0.0 : 1.0 / (z + xi * r);
      const float mf2 = mf * mf;
      const float a1 = fy * mf;
      const float a2 = fy * xi * y * y / r * mf2;
      const float a3 = fy * xi * x * y / r * mf2;
      const float a4 = fy * y * (xi * z / r + 1.0) * mf2;

      Jj[0] = -h * a3;
      Jj[1] = h * (a1 - a2);
      Jj[2] = -h * a4;

      Jj[3] = -z * (a1 - a2) - y * a4;
      Jj[4] = -z * a3 + x * a4;
      Jj[5] = y * a3 + x * (a1 - a2);

      Jz = -tij[0] * a3 + tij[1] * (a1 - a2) 
           - tij[2] * a4;
    }

    Cii[block_id][k] += wv * Jz * Jz;
    bz[block_id][k] += wv * rv * Jz;

    if (ix == jx) wv = 0;

    adjSE3(tij, qij, Jj, Ji);
    for (int n=0; n<6; n++) Ji[n] *= -1;

    l=0;
    for (int n=0; n<12; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += wv * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<6; n++) {
      vi[n] += wv * rv * Ji[n];
      vj[n] += wv * rv * Jj[n];

      Eii[block_id][n][k] += wv * Jz * Ji[n];
      Eij[block_id][n][k] += wv * Jz * Jj[n];
    }
    

    if (opt_intr){
      // calib jacobian pinhole
      if (model_id == 0){
        Jc[0] = -R[1][0] * (u-cx) * fy * d * (1.0/fx) * (1.0/fx) 
                + fy * y * d2 * R[2][0] * (u-cx) * (1.0/fx) * (1.0/fx);
        Jc[1] = -R[1][1] * (v-cy) * (1.0/fy) * d 
                + R[2][1] * (v-cy) * (1.0/fy) * y * d2 + y * d;
        Jc[2] = -R[1][0] * fy * (1.0/fx) * d + fy * y * d2 * R[2][0] * (1.0/fx);
        Jc[3] = -R[1][1] * d + y * d2 * R[2][1] + 1.0;
      }
      else if (model_id == 2){
        // calib jacobian focal
        float zi = 1;
        float a1 =  (u-cx) * zi * (1.0/fx) * (1.0/fx);
        float a2 =  (v-cy) * zi * (1.0/fy) * (1.0/fy);
        float l1 = - R[0][0] * a1 - R[0][1] * a2;
        float l2 = - R[1][0] * a1 - R[1][1] * a2;
        float l3 = - R[2][0] * a1 - R[2][1] * a2;
        Jc[0] = fx * d * l2 - fx * y * d2 * l3 + y * d;
      }
      else{
        // calib jacobian mei model
        const float mu = 0.01;
        const float disp = disps[ix][i][j];
        const int component = 1;

        for (int direction = 0; direction < 5; direction++) {
            finite_difference_mei(fx, fy, cx, cy, xi, uvi, qij, tij, disp, 
                                  mu, direction, component, Jc);
        }
      }

      l=0;
      for (int n=0; n<n_intr; n++) {
        for (int m=0; m<=n; m++) {
          cij[l] += wv * Jc[n] * Jc[m];
          l++;
        }
      }

      // calib-pose block
      for (int n=0; n<n_intr; n++) {
        for (int m=0; m<6; m++) {
          cpij[0][n][m] += wv * Jc[n] * Ji[m];
          cpij[1][n][m] += wv * Jc[n] * Jj[m];
        }
      }

      for (int n=0; n<n_intr; n++) {
        q[n] += wv * rv * Jc[n];
        CalibDepth[block_id][k][n] += wv * Jz * Jc[n];
      }

    }
  }



  __syncthreads();

  // sum results of all threads and write to output array
  __shared__ float sdata[THREADS];
  for (int n=0; n<6; n++) {
    sdata[threadIdx.x] = vi[n];
    // blockReduce sums all entries of sdata efficiently
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      vs[0][block_id][n] = sdata[0];
    }

    __syncthreads();

    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      vs[1][block_id][n] = sdata[0];
    }

  }

  l=0;
  for (int n=0; n<12; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);

      if (threadIdx.x == 0) {
        if (n<6 && m<6) {
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];
        }
        else if (n >=6 && m<6) {
          Hs[1][block_id][m][n-6] = sdata[0];
          Hs[2][block_id][n-6][m] = sdata[0];
        }
        else {
          Hs[3][block_id][n-6][m-6] = sdata[0];
          Hs[3][block_id][m-6][n-6] = sdata[0];
        }
      }

      l++;
    }
  }
  

  if (opt_intr){

    __syncthreads();
    l=0;
    for (int n=0; n<n_intr; n++) {
      for (int m=0; m<=n; m++) {
        sdata[threadIdx.x] = cij[l];
        blockReduce(sdata);

        if (threadIdx.x == 0) {
          Calib[block_id][n][m] = sdata[0];
          Calib[block_id][m][n] = sdata[0];
        }
        l++;
      }
    }

    __syncthreads();
    for (int l = 0; l<2; l++){
      for (int n=0; n<n_intr; n++) {
        for (int m=0; m<6; m++) {
          sdata[threadIdx.x] = cpij[l][n][m];
          blockReduce(sdata);

          if (threadIdx.x == 0) {
            CalibPose[block_id][l][n][m] = sdata[0];
          }
        }
      }
    }

    __syncthreads();

    for (int n=0; n<n_intr; n++) {
      sdata[threadIdx.x] = q[n];
      blockReduce(sdata);
      if (threadIdx.x == 0) {
        qs[block_id][n] = sdata[0];
      }
    }
    __syncthreads();
  }

}


__global__ void projmap_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> valid)
{

  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ int ix;
  __shared__ int jx;

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    ix = static_cast<int>(ii[block_id]);
    jx = static_cast<int>(jj[block_id]);
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
  }

  __syncthreads();

  // load poses from global memory
  if (thread_id < 3) {
    ti[thread_id] = poses[ix][thread_id];
    tj[thread_id] = poses[jx][thread_id];
  }

  if (thread_id < 4) {
    qi[thread_id] = poses[ix][thread_id+3];
    qj[thread_id] = poses[jx][thread_id+3];
  }

  __syncthreads();

  if (thread_id == 0) {
    relSE3(ti, qi, tj, qj, tij, qij);
  }

  //points 
  float Xi[4];
  float Xj[4];

  __syncthreads();

  GPU_1D_KERNEL_LOOP(k, ht*wd) {
    const int i = k / wd;
    const int j = k % wd;

    const float u = static_cast<float>(j);
    const float v = static_cast<float>(i);
    
    // homogenous coordinates
    Xi[0] = (u - cx) / fx;
    Xi[1] = (v - cy) / fy;
    Xi[2] = 1;
    Xi[3] = disps[ix][i][j];

    // transform homogenous point
    actSE3(tij, qij, Xi, Xj);

    coords[block_id][i][j][0] = u;
    coords[block_id][i][j][1] = v;

    if (Xj[2] > 0.01) {
      coords[block_id][i][j][0] = fx * (Xj[0] / Xj[2]) + cx;
      coords[block_id][i][j][1] = fy * (Xj[1] / Xj[2]) + cy;
    }

    valid[block_id][i][j][0] = (Xj[2] > MIN_DEPTH) ? 1.0 : 0.0;

  }
}

__global__ void frame_distance_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> dist,
    const float beta,
    const int model_id) {

  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ int ix;
  __shared__ int jx;

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;
  __shared__ float xi;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    ix = static_cast<int>(ii[block_id]);
    jx = static_cast<int>(jj[block_id]);
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
    if (model_id == 1){
      xi = intrinsics[4];
    }
  }

  __syncthreads();

  //points 
  float Xi[4];
  float Xj[4];

  __shared__ float accum[THREADS]; accum[thread_id] = 0;
  __shared__ float valid[THREADS]; valid[thread_id] = 0;
  __shared__ float total[THREADS]; total[thread_id] = 0;

  __syncthreads();

  for (int n=0; n<1; n++) {

    if (thread_id < 3) {
      ti[thread_id] = poses[ix][thread_id];
      tj[thread_id] = poses[jx][thread_id];
    }

    if (thread_id < 4) {
      qi[thread_id] = poses[ix][thread_id+3];
      qj[thread_id] = poses[jx][thread_id+3];
    }

    __syncthreads();


    relSE3(ti, qi, tj, qj, tij, qij);

    float d, du, dv;

    GPU_1D_KERNEL_LOOP(k, ht*wd) {
      const int i = k / wd;
      const int j = k % wd;

      const float u = static_cast<float>(j);
      const float v = static_cast<float>(i);


      
      if (model_id == 0 || model_id == 2){
        // homogenous coordinates
        Xi[0] = (u - cx) / fx;
        Xi[1] = (v - cy) / fy;
        Xi[2] = 1;
        Xi[3] = disps[ix][i][j];

        // transform homogenous point
        actSE3(tij, qij, Xi, Xj);

        du = fx * (Xj[0] / Xj[2]) + cx - u;
        dv = fy * (Xj[1] / Xj[2]) + cy - v;
        }

      else{
        const float xs = (u - cx) / fx;
        const float ys = (v - cy) / fy;
        const float beta = (xi + sqrt( 1 + ( 1 - xi * xi) * (xs * xs  + ys * ys)) ) 
                           / (1 + xs * xs  + ys * ys);

        Xi[0] = beta * xs / (beta - xi);
        Xi[1] = beta * ys / (beta - xi);
        Xi[2] = 1.0;
        Xi[3] = disps[ix][i][j];

        // transform homogenous point
        actSE3(tij, qij, Xi, Xj);

        const float d = (Xj[2] < MIN_DEPTH) ? 0.0 : 1.0 / Xj[2];
        const float z = (Xj[2] < MIN_DEPTH) ? 0.0 : Xj[2];
        const float r = sqrt(Xj[0] * Xj[0] + Xj[1] * Xj[1] + z * z);
        const float mei_factor = ((z + xi * r) < MIN_DEPTH) ? 0.0 : 1.0 / (z + xi * r);

        du = fx * Xj[0] * mei_factor + cx - u;
        dv = fy * Xj[1] * mei_factor + cy - v;
      }

      d = sqrtf(du*du + dv*dv);

      total[threadIdx.x] += beta;
      
      if (Xj[2] > MIN_DEPTH) {
        accum[threadIdx.x] += beta * d;
        valid[threadIdx.x] += beta;
      }

      if (model_id == 0 || model_id == 2){
        Xi[0] = (u - cx) / fx;
        Xi[1] = (v - cy) / fy;
        Xi[2] = 1;
        Xi[3] = disps[ix][i][j];

        Xj[0] = Xi[0] + Xi[3] * tij[0];
        Xj[1] = Xi[1] + Xi[3] * tij[1];
        Xj[2] = Xi[2] + Xi[3] * tij[2];

        du = fx * (Xj[0] / Xj[2]) + cx - u;
        dv = fy * (Xj[1] / Xj[2]) + cy - v;
        }
      else{
        const float xs = (u - cx) / fx;
        const float ys = (v - cy) / fy;
        const float beta = (xi + sqrt( 1 + ( 1 - xi * xi) * (xs * xs  + ys * ys)) ) 
                           / (1 + xs * xs  + ys * ys);

        Xi[0] = beta * xs / (beta - xi);
        Xi[1] = beta * ys / (beta - xi);
        Xi[2] = 1.0;
        Xi[3] = disps[ix][i][j];

        Xj[0] = Xi[0] + Xi[3] * tij[0];
        Xj[1] = Xi[1] + Xi[3] * tij[1];
        Xj[2] = Xi[2] + Xi[3] * tij[2];

        const float d = (Xj[2] < MIN_DEPTH) ? 0.0 : 1.0 / Xj[2];
        const float z = (Xj[2] < MIN_DEPTH) ? 0.0 : Xj[2];
        const float r = sqrt(Xj[0] * Xj[0] + Xj[1] * Xj[1] + z * z);
        const float mei_factor = ((z + xi * r) < MIN_DEPTH) ? 0.0 : 1.0 / (z + xi * r);

        du = fx * Xj[0] * mei_factor + cx - u;
        dv = fy * Xj[1] * mei_factor + cy - v;
        }

      d = sqrtf(du*du + dv*dv);

      total[threadIdx.x] += (1 - beta);
      
      if (Xj[2] > MIN_DEPTH) {
        accum[threadIdx.x] += (1 - beta) * d;
        valid[threadIdx.x] += (1 - beta);
      }
    }

    if (threadIdx.x == 0) {
      int tmp = ix;
      ix = jx;
      jx = tmp;
    }

    __syncthreads();

  }
  __syncthreads(); blockReduce(accum);
  __syncthreads(); blockReduce(total);
  __syncthreads(); blockReduce(valid);

  __syncthreads();

  if (thread_id == 0) {
    dist[block_id] = (valid[0] / (total[0] + 1e-8) < 0.75) ? 1000.0 : accum[0] / valid[0];
  }
}



__global__ void depth_filter_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> thresh,
    const int model_id,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> counter
    )
{

  const int block_id = blockIdx.x;
  const int neigh_id = blockIdx.y;
  const int index = blockIdx.z * blockDim.x + threadIdx.x;

  // if (threadIdx.x == 0) {
  //   printf("%d %d %d %d\n", blockIdx.x, blockIdx.y, blockDim.x, threadIdx.x);
  // }

  const int num = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ int ix;
  __shared__ int jx;

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;
  __shared__ float xi;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  if (threadIdx.x == 0) {
    ix = static_cast<int>(inds[block_id]);
    jx = (neigh_id < 3) ? ix - neigh_id - 1 : ix + neigh_id;
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
    if (model_id == 1){
      xi = intrinsics[4];
    }
  }

  __syncthreads();

  if (jx < 0 || jx >= num) {
    return;
  }

  const float t = thresh[block_id];

  // load poses from global memory
  if (threadIdx.x < 3) {
    ti[threadIdx.x] = poses[ix][threadIdx.x];
    tj[threadIdx.x] = poses[jx][threadIdx.x];
  }

  if (threadIdx.x < 4) {
    qi[threadIdx.x] = poses[ix][threadIdx.x+3];
    qj[threadIdx.x] = poses[jx][threadIdx.x+3];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    relSE3(ti, qi, tj, qj, tij, qij);
  }

  //points 
  float Xi[4];
  float Xj[4];
  float uj, vj, dj;

  __syncthreads();

  if (index < ht*wd) {
    const int i = index / wd;
    const int j = index % wd;

    const float ui = static_cast<float>(j);
    const float vi = static_cast<float>(i);
    const float di = disps[ix][i][j];
    

    if (model_id == 0 || model_id == 2){
      // homogenous coordinates
      Xi[0] = (ui - cx) / fx;
      Xi[1] = (vi - cy) / fy;
      Xi[2] = 1;
      Xi[3] = di;

      // transform homogenous point
      actSE3(tij, qij, Xi, Xj);

      uj = fx * (Xj[0] / Xj[2]) + cx;
      vj = fy * (Xj[1] / Xj[2]) + cy;
      dj = Xj[3] / Xj[2];
    }
    else{
      // homogenous coordinates
      const float xs = (ui - cx) / fx;
      const float ys = (vi - cy) / fy;
      const float beta = (xi + sqrt( 1 + ( 1 - xi * xi) * (xs * xs  + ys * ys)) ) 
                         / (1 + xs * xs  + ys * ys);

      Xi[0] = beta * xs / (beta - xi);
      Xi[1] = beta * ys / (beta - xi);
      Xi[2] = 1.0;
      Xi[3] = di;

      // transform homogenous point
      actSE3(tij, qij, Xi, Xj);

      // const float d = (Xj[2] < MIN_DEPTH) ? 0.0 : 1.0 / Xj[2];
      const float z = (Xj[2] < MIN_DEPTH) ? 0.0 : Xj[2]; 
      const float r = sqrt(Xj[0] * Xj[0] + Xj[1] * Xj[1] + z * z);
      const float mei_factor = ((z + xi * r) < MIN_DEPTH) ? 0.0 : 1.0 / (z + xi * r);

      uj = fx * Xj[0] * mei_factor + cx;
      vj = fy * Xj[1] * mei_factor + cy;
      dj = Xj[3] / Xj[2]; 

    }

    const int u0 = static_cast<int>(floor(uj));
    const int v0 = static_cast<int>(floor(vj));

    if (u0 >= 0 && v0 >= 0 && u0 < wd-1 && v0 < ht-1) {
      const float wx = ceil(uj) - uj;
      const float wy = ceil(vj) - vj;

      const float d00 = disps[jx][v0+0][u0+0];
      const float d01 = disps[jx][v0+0][u0+1];
      const float d10 = disps[jx][v0+1][u0+0];
      const float d11 = disps[jx][v0+1][u0+1];

      const float dj_hat = wy*wx*d00 + wy*(1-wx)*d01 + (1-wy)*wx*d10 + (1-wy)*(1-wx)*d11;

      const float err = abs(1.0/dj - 1.0/dj_hat);
      if       (abs(1.0/dj - 1.0/d00) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
      else if  (abs(1.0/dj - 1.0/d01) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
      else if  (abs(1.0/dj - 1.0/d10) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
      else if  (abs(1.0/dj - 1.0/d11) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
    }
  }
}



__global__ void iproj_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const int model_id,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> points)

{

  const int block_id = blockIdx.x;
  const int index = blockIdx.y * blockDim.x + threadIdx.x;


  const int num = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;
  __shared__ float xi;

  __shared__ float t[3];
  __shared__ float q[4];

  if (threadIdx.x == 0) {
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
    if (model_id == 1){
      xi = intrinsics[4];
    }
  }

  __syncthreads();


  // load poses from global memory
  if (threadIdx.x < 3) {
    t[threadIdx.x] = poses[block_id][threadIdx.x];
  }

  if (threadIdx.x < 4) {
    q[threadIdx.x] = poses[block_id][threadIdx.x+3];
  }

  __syncthreads();

  // points 
  float Xi[4];
  float Xj[4];

  if (index < ht*wd) {
    const int i = index / wd;
    const int j = index % wd;

    const float ui = static_cast<float>(j);
    const float vi = static_cast<float>(i);
    const float di = disps[block_id][i][j];
    
    if (model_id == 0 || model_id == 2){
      // homogenous coordinates
      Xi[0] = (ui - cx) / fx;
      Xi[1] = (vi - cy) / fy;
      Xi[2] = 1;
      Xi[3] = di;
    }
    else{
      const float xs = (ui - cx) / fx;
      const float ys = (vi - cy) / fy;
      const float beta = (xi + sqrt( 1 + ( 1 - xi * xi) * (xs * xs  + ys * ys)) ) 
                          / (1 + xs * xs  + ys * ys);

      Xi[0] = beta * xs / (beta - xi);
      Xi[1] = beta * ys / (beta - xi);
      Xi[2] = 1.0;
      Xi[3] = di;
    }

    // transform homogenous point
    actSE3(t, q, Xi, Xj);

    points[block_id][i][j][0] = Xj[0] / Xj[3];
    points[block_id][i][j][1] = Xj[1] / Xj[3];
    points[block_id][i][j][2] = Xj[2] / Xj[3];

  }
}



__global__ void accum_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> inps,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ptrs,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> idxs,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> outs)
{
  
  const int block_id = blockIdx.x;
  const int D = inps.size(2);

  const int start = ptrs[block_id];
  const int end = ptrs[block_id+1];

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    float x = 0;
    for (int i=start; i<end; i++) {
      x += inps[idxs[i]][k];
    }
    outs[block_id][k] = x;
  }  
}


__device__ void
retrSE3(const float *xi, const float* t, const float* q, float* t1, float* q1) {
  // retraction on SE3 manifold

  float dt[3] = {0, 0, 0};
  float dq[4] = {0, 0, 0, 1};
  
  expSE3(xi, dt, dq);

  q1[0] = dq[3] * q[0] + dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1];
  q1[1] = dq[3] * q[1] + dq[1] * q[3] + dq[2] * q[0] - dq[0] * q[2];
  q1[2] = dq[3] * q[2] + dq[2] * q[3] + dq[0] * q[1] - dq[1] * q[0];
  q1[3] = dq[3] * q[3] - dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2];

  actSO3(dq, t, t1);
  t1[0] += dt[0];
  t1[1] += dt[1];
  t1[2] += dt[2];
}


__global__ void pose_retr_kernel(
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dx,
    const int t0, const int t1) 
{

  for (int k=t0+threadIdx.x; k<t1; k+=blockDim.x) {
    float xi[6], q[4], q1[4], t[3], t1[3];

    t[0] = poses[k][0];
    t[1] = poses[k][1];
    t[2] = poses[k][2];

    q[0] = poses[k][3];
    q[1] = poses[k][4];
    q[2] = poses[k][5];
    q[3] = poses[k][6];
    
    for (int n=0; n<6; n++) {
      xi[n] = dx[k-t0][n];
    }

    retrSE3(xi, t, q, t1, q1);

    poses[k][0] = t1[0];
    poses[k][1] = t1[1];
    poses[k][2] = t1[2];

    poses[k][3] = q1[0];
    poses[k][4] = q1[1];
    poses[k][5] = q1[2];
    poses[k][6] = q1[3];
  }
}

__global__ void disp_retr_kernel(
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dz,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> inds) 
{
  const int i = inds[blockIdx.x];
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  for (int k=threadIdx.x; k<ht*wd; k+=blockDim.x) {
    float d = disps[i][k/wd][k%wd] + dz[blockIdx.x][k];
    disps[i][k/wd][k%wd] = d;
  }
}

torch::Tensor accum_cuda(torch::Tensor data, torch::Tensor ix, torch::Tensor jx) {
  torch::Tensor ix_cpu = ix.to(torch::kCPU);
  torch::Tensor jx_cpu = jx.to(torch::kCPU);
  torch::Tensor inds = torch::argsort(ix_cpu);

  long* ix_data = ix_cpu.data_ptr<long>();
  long* jx_data = jx_cpu.data_ptr<long>();
  long* kx_data = inds.data_ptr<long>();

  int count = jx.size(0);
  std::vector<int> cols;

  torch::Tensor ptrs_cpu = torch::zeros({count+1}, 
    torch::TensorOptions().dtype(torch::kInt64));
  
  long* ptrs_data = ptrs_cpu.data_ptr<long>();
  ptrs_data[0] = 0;

  int i = 0;
  for (int j=0; j<count; j++) {
    while (i < ix.size(0) && ix_data[kx_data[i]] <= jx_data[j]) {
      if (ix_data[kx_data[i]] == jx_data[j])
        cols.push_back(kx_data[i]);
      i++;
    }
    ptrs_data[j+1] = cols.size();
  }

  torch::Tensor idxs_cpu = torch::zeros({long(cols.size())}, 
    torch::TensorOptions().dtype(torch::kInt64));

  long* idxs_data = idxs_cpu.data_ptr<long>();

  for (int i=0; i<cols.size(); i++) {
    idxs_data[i] = cols[i];
  }

  torch::Tensor ptrs = ptrs_cpu.to(torch::kCUDA);
  torch::Tensor idxs = idxs_cpu.to(torch::kCUDA);

  torch::Tensor out = torch::zeros({jx.size(0), data.size(1)},
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  accum_kernel<<<count, THREADS>>>(
    data.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    ptrs.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    idxs.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    out.packed_accessor32<float,2,torch::RestrictPtrTraits>());

  return out;
}


__global__ void EEt6x6_kernel(
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> E,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Q,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> S)
{

  // indicices
  const int ix = idx[blockIdx.x][0];
  const int jx = idx[blockIdx.x][1];
  const int kx = idx[blockIdx.x][2];

  const int D = E.size(2);

  float dS[6][6];
  float ei[6];
  float ej[6];

  for (int i=0; i<6; i++) {
    for (int j=0; j<6; j++) {
      dS[i][j] = 0;
    }
  }

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    const float q = Q[kx][k];
      
    // coalesced memory read
    for (int n=0; n<6; n++) {
      ei[n] = E[ix][n][k] * q;
      ej[n] = E[jx][n][k];
    }

    // block EEt
    for (int n=0; n<6; n++) {
      for (int m=0; m<6; m++) {
        dS[n][m] += ei[n] * ej[m];
      }
    }
  }

  __syncthreads();
  __shared__ float sdata[THREADS];

  for (int n=0; n<6; n++) {
    for (int m=0; m<6; m++) {
      sdata[threadIdx.x] = dS[n][m];

      blockReduce(sdata);

      if (threadIdx.x == 0) {
        S[blockIdx.x][n][m] = sdata[0];
      }
    }
  }
}

__global__ void  ECDT_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> E,
    const torch::PackedTensorAccessor32<float, 3,torch::RestrictPtrTraits> CD, 
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Q,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> ECDT_ext)
  {
  const int D = E.size(2); 
  const int n_intr = CD.size(1);
  const int kx = idx[blockIdx.x][0];

  float b[6][5]; 

  for (int n=0; n<6; n++) {
    for (int m=0; m<n_intr; m++) {
      b[n][m] = 0.0;
    }
  }


  for (int k=threadIdx.x; k<D; k+=blockDim.x) {

    for (int m=0; m<n_intr; m++) {
      const float q_w = Q[kx][k] * CD[kx][m][k];

      for (int n=0; n<6; n++) {
        b[n][m] += q_w * E[blockIdx.x][n][k];
      }
    }
  }

  __syncthreads();
  __shared__ float sdata[THREADS];

  for (int n=0; n<6; n++) {
    for (int m=0; m<n_intr; m++) {

      sdata[threadIdx.x] = b[n][m];
      blockReduce(sdata);

      if (threadIdx.x == 0) {
        ECDT_ext[blockIdx.x][n][m] += sdata[0];
      }
    }
  }
}



__global__ void Ev6x1_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> E,
    const torch::PackedTensorAccessor32<float, 2,torch::RestrictPtrTraits> Q,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> w,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> v)
{
  const int D = E.size(2);
  const int kx = idx[blockIdx.x][0];

  float b[6];
  for (int n=0; n<6; n++) {
    b[n] = 0.0;
  }

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    const float q_w = Q[kx][k] * w[kx][k];

    for (int n=0; n<6; n++) {
      b[n] += q_w * E[blockIdx.x][n][k];
    }
  }

  __syncthreads();
  __shared__ float sdata[THREADS];

  for (int n=0; n<6; n++) {
    sdata[threadIdx.x] = b[n];
    blockReduce(sdata);

    if (threadIdx.x == 0) {
      v[blockIdx.x][n] += sdata[0];
    }
  }
}

__global__ void EvT6x1_kernel(
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> E, 
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> x, 
  const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> idx, 
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> w) 
{

  const int D = E.size(2);
  const int ix = idx[blockIdx.x];

  if (idx[blockIdx.x] <= 0 || idx[blockIdx.x] >= x.size(0))
    return;

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    float dw = 0;
    for (int n=0; n<6; n++) {
      dw += E[blockIdx.x][n][k] * x[ix][n];
    }
    w[blockIdx.x][k] = dw;
  }
}

__global__ void CDCDT_kernel(
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> CD, 
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Q, 
  torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> CDCDT) 
{
  const int D = CD.size(2); 
  const int n_intr = CD.size(1);
  float v_thread[5][5]; 


  for (int n=0; n<n_intr; n++) {
    for (int m=0; m<=n; m++) {
      float val = 0;

      for (int k=threadIdx.x; k<D; k+=blockDim.x) {
        val += CD[blockIdx.x][n][k] * Q[blockIdx.x][k] * CD[blockIdx.x][m][k];
      }
      v_thread[n][m] = val;
      v_thread[m][n] = val;
    }
  }

  __syncthreads();
  __shared__ float sdata[THREADS];

  for (int n=0; n<n_intr; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = v_thread[n][m];
      blockReduce(sdata);

      if (threadIdx.x == 0) {
        CDCDT[blockIdx.x][n][m] = sdata[0];
        CDCDT[blockIdx.x][m][n] = sdata[0];
      }
    }
  }

}

__global__ void CDv_kernel(
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> CD, 
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Q, 
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> w, 
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> v_ext) 

{
  const int D = CD.size(2); 
  const int n_intr = CD.size(1);
  float v_thread[5];

  for (int n=0; n<n_intr; n++) {
    float v_exti = 0;

    for (int k=threadIdx.x; k<D; k+=blockDim.x) {
      v_exti += CD[blockIdx.x][n][k] * Q[blockIdx.x][k] * w[blockIdx.x][k];
    }
    v_thread[n] = v_exti;
  }

  __syncthreads();
  __shared__ float sdata[THREADS];

  for (int n=0; n<n_intr; n++) {
    sdata[threadIdx.x] = v_thread[n];
    blockReduce(sdata);

    if (threadIdx.x == 0) {
      v_ext[blockIdx.x][n] += sdata[0];
    }
  }


}

__global__ void CDvT_kernel(
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> CD, 
  const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> dI, 
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dw_add) 
{

  const int D = CD.size(2); 
  const int n_intr = CD.size(1);

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    float dw = 0;
    for (int n=0; n<n_intr; n++) {
      dw += CD[blockIdx.x][n][k] * dI[n];
    }
    dw_add[blockIdx.x][k] = dw;
  }
  __syncthreads();
}


torch::Tensor accum_CalibPose(torch::Tensor CPii, torch::Tensor ii, const int n_unique) {
  
  const int n_intr = CPii.size(1);

  auto CPii_cpu = CPii.to(torch::kCPU).to(torch::kFloat64);
  auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);

  auto CPii_acc = CPii_cpu.accessor<double,3>();
  auto ii_acc = ii_cpu.accessor<long,1>();

  torch::Tensor CP = torch::zeros({n_unique*6, n_intr});
  
  for (int p=0; p<n_intr; p++) {

    for (int n=0; n<ii.size(0); n++) {
      const int i = ii_acc[n];
      if (i >= 0) {
        for (int j=0; j<6; j++) {
          CP[i*6 + j][p] += CPii_acc[n][p][j];
        }
      }
    }
  }

  return CP;
}



class SparseBlock {
  public:

    Eigen::SparseMatrix<double> A;
    Eigen::VectorX<double> b;

    SparseBlock(int N, int M) : N(N), M(M) {
      A = Eigen::SparseMatrix<double>(N*M, N*M);
      b = Eigen::VectorXd::Zero(N*M);
    }

    SparseBlock(Eigen::SparseMatrix<double> const& A, Eigen::VectorX<double> const& b, 
        int N, int M) : A(A), b(b), N(N), M(M) {}

    void update_lhs(torch::Tensor As, torch::Tensor ii, torch::Tensor jj) {

      auto As_cpu = As.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);
      auto jj_cpu = jj.to(torch::kCPU).to(torch::kInt64);

      auto As_acc = As_cpu.accessor<double,3>();
      auto ii_acc = ii_cpu.accessor<long,1>();
      auto jj_acc = jj_cpu.accessor<long,1>();

      std::vector<T> tripletList;
      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        const int j = jj_acc[n];

        if (i >= 0 && j >= 0) {
          for (int k=0; k<M; k++) {
            for (int l=0; l<M; l++) {
              double val = As_acc[n][k][l];
              tripletList.push_back(T(M*i + k, M*j + l, val));
            }
          }
        }
      }
      A.setFromTriplets(tripletList.begin(), tripletList.end());
    }

    void update_rhs(torch::Tensor bs, torch::Tensor ii) {
      auto bs_cpu = bs.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);

      auto bs_acc = bs_cpu.accessor<double,2>();
      auto ii_acc = ii_cpu.accessor<long,1>();

      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        if (i >= 0) {
          for (int j=0; j<M; j++) {
            b(i*M + j) += bs_acc[n][j];
          }
        }
      }
    }

    SparseBlock operator-(const SparseBlock& S) {
      return SparseBlock(A - S.A, b - S.b, N, M);
    }

    std::tuple<torch::Tensor, torch::Tensor> get_dense() {
      Eigen::MatrixXd Ad = Eigen::MatrixXd(A);

      torch::Tensor H = torch::from_blob(Ad.data(), {N*M, N*M}, torch::TensorOptions()
        .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);

      torch::Tensor v = torch::from_blob(b.data(), {N*M, 1}, torch::TensorOptions()
        .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);

      return std::make_tuple(H, v);

    }

    torch::Tensor solve(const float lm=0.0001, const float ep=0.1) {

      torch::Tensor dx;

      Eigen::SparseMatrix<double> L(A);
      L.diagonal().array() += ep + lm * L.diagonal().array();

      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
      solver.compute(L); // Compute the sparse Cholesky decomposition of matrix L

      if (solver.info() == Eigen::Success) {
        Eigen::VectorXd x = solver.solve(b);
        dx = torch::from_blob(x.data(), {N, M}, torch::TensorOptions()
          .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);
      }
      else {
        dx = torch::zeros({N, M}, torch::TensorOptions()
          .device(torch::kCUDA).dtype(torch::kFloat32));
      }
      
      return dx;
    }

  private:
    const int N;
    const int M;

};



class SparseBlockAsym {
  public:

    Eigen::SparseMatrix<double> A;
    Eigen::VectorX<double> b;

    SparseBlockAsym(int N) : N(N) {
      A = Eigen::SparseMatrix<double>(N, N);
      b = Eigen::VectorXd::Zero(N);
    }

    SparseBlockAsym(Eigen::SparseMatrix<double> const& A, Eigen::VectorX<double> const& b, 
        int N) : A(A), b(b), N(N){}

    void update_lhs_sym(torch::Tensor As, torch::Tensor ii, torch::Tensor jj, const int M) {

      auto As_cpu = As.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);
      auto jj_cpu = jj.to(torch::kCPU).to(torch::kInt64);

      auto As_acc = As_cpu.accessor<double,3>();
      auto ii_acc = ii_cpu.accessor<long,1>();
      auto jj_acc = jj_cpu.accessor<long,1>();

      std::vector<T> tripletList;
      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        const int j = jj_acc[n];

        if (i >= 0 && j >= 0) {
          for (int k=0; k<M; k++) {
            for (int l=0; l<M; l++) {
              double val = As_acc[n][k][l];
              tripletList.push_back(T(M*i + k, M*j + l, val));
            }
          }
        }
      }
      A.setFromTriplets(tripletList.begin(), tripletList.end());
    }

    void update_lhs_extended(torch::Tensor As, torch::Tensor ii, torch::Tensor jj, 
                             const int M, std::vector<T> extTripletList) 
    {

      auto As_cpu = As.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);
      auto jj_cpu = jj.to(torch::kCPU).to(torch::kInt64);

      auto As_acc = As_cpu.accessor<double,3>();
      auto ii_acc = ii_cpu.accessor<long,1>();
      auto jj_acc = jj_cpu.accessor<long,1>();

      std::vector<T> tripletList;
      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        const int j = jj_acc[n];

        if (i >= 0 && j >= 0) {
          for (int k=0; k<M; k++) {
            for (int l=0; l<M; l++) {
              double val = As_acc[n][k][l];
              tripletList.push_back(T(M*i + k, M*j + l, val));
            }
          }
        }
      }
      

      for (int iext=0; iext < extTripletList.size(); iext++){
        tripletList.push_back(extTripletList[iext]);
      }

      A.setFromTriplets(tripletList.begin(), tripletList.end());
    }

    void update_rhs_sym(torch::Tensor bs, torch::Tensor ii, const int M) {
      auto bs_cpu = bs.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);

      auto bs_acc = bs_cpu.accessor<double,2>();
      auto ii_acc = ii_cpu.accessor<long,1>();

      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        if (i >= 0) {
          for (int j=0; j<M; j++) {
            b(i*M + j) += bs_acc[n][j];
          }
        }
      }
    }

    void update_rhs_extended(torch::Tensor bs, torch::Tensor ii, 
                             const int M, torch::Tensor ext_vec) {

      auto bs_cpu = bs.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);
      auto ext_cpu = ext_vec.to(torch::kCPU).to(torch::kFloat64);

      auto bs_acc = bs_cpu.accessor<double,2>();
      auto ii_acc = ii_cpu.accessor<long,1>();
      auto ext_acc = ext_cpu.accessor<double,1>();

      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        if (i >= 0) {
          for (int j=0; j<M; j++) {
            b(i*M + j) += bs_acc[n][j];
          }
        }
      }
      
      for (int n=0; n<ext_acc.size(0); n++){
        b(N - ext_acc.size(0) + n) = ext_acc[n];
      }


    }
    SparseBlockAsym operator-(const SparseBlockAsym& S) {
      return SparseBlockAsym(A - S.A, b - S.b, N);
    }


    torch::Tensor solve(const float lm=0.0001, const float ep=0.1) {

      torch::Tensor dx;

      Eigen::SparseMatrix<double> L(A);
      L.diagonal().array() += ep + lm * L.diagonal().array();

      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
      solver.compute(L);

      if (solver.info() == Eigen::Success) {
        Eigen::VectorXd x = solver.solve(b);
        dx = torch::from_blob(x.data(), {N}, torch::TensorOptions()
          .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);
      }
      else {
        std::cout << "Warning: Cholesky decomposition was not successful." << std::endl;
        dx = torch::zeros({N}, torch::TensorOptions()
          .device(torch::kCUDA).dtype(torch::kFloat32));
      }
      
      return dx;
    }

  private:
    const int N;

};


SparseBlockAsym pose_calib_block(torch::Tensor Hs,
                        torch::Tensor vs,
                        torch::Tensor Calib,
                        torch::Tensor q,
                        torch::Tensor CalibPose,
                        torch::Tensor ii,
                        torch::Tensor jj,
                        const int t0,
                        const int t1){
  

  torch::Tensor C_block = torch::sum(Calib, 0); 
  const int n_intr = C_block.size(0);
  torch::Tensor q_vec = torch::sum(q, 0); 
  int n_unique = t1-t0; 

  torch::Tensor CPii = CalibPose.permute({1, 0, 2, 3}).reshape({-1, n_intr, 6}); 
  torch::Tensor CP = accum_CalibPose(CPii, torch::cat({ii, jj}) - t0, n_unique); 

  // get triplet list to extend pose hessian with calib blocks
  int i0 = n_unique * 6;
  std::vector<T> extTripletList;

  // calib block
  for (int i=0; i<n_intr; i++){
    for (int j=0; j<n_intr; j++){
      double val = C_block[i][j].item<double>();
      extTripletList.push_back(T(i0+i, i0+j, val));
    }
  }

  // calib-pose block
  for (int i=0; i<n_intr; i++){
    for (int j=0; j<i0; j++){
      double val = CP[j][i].item<double>();
      extTripletList.push_back(T(i0+i, j, val));
      extTripletList.push_back(T(j, i0+i, val));
    }
  }

  // pose-calib block
  SparseBlockAsym A(n_unique * 6 + n_intr);
  int M = 6;

  A.update_lhs_extended(Hs.reshape({-1, 6, 6}), 
      torch::cat({ii, ii, jj, jj}) - t0, 
      torch::cat({ii, jj, ii, jj}) - t0,
      M,
      extTripletList);

  A.update_rhs_extended(vs.reshape({-1, 6}), 
      torch::cat({ii, jj}) - t0, M, q_vec);

  return A;
}



SparseBlockAsym schur_calib_block(torch::Tensor E, 
                        torch::Tensor Q, 
                        torch::Tensor CD, 
                        torch::Tensor w, 
                        torch::Tensor ii, 
                        torch::Tensor jj, 
                        torch::Tensor kk, 
                        torch::Tensor kx, 
                        const int t0,
                        const int t1)
{

  torch::Tensor ii_cpu = ii.to(torch::kCPU);
  torch::Tensor jj_cpu = jj.to(torch::kCPU);
  torch::Tensor kk_cpu = kk.to(torch::kCPU);

  const int P = t1 - t0;
  const long* ii_data = ii_cpu.data_ptr<long>();
  const long* jj_data = jj_cpu.data_ptr<long>();
  const long* kk_data = kk_cpu.data_ptr<long>();
  const int ht_wd = E.size(2);
  const int n_intr = CD.size(1);

  std::vector<std::vector<long>> graph(P);
  std::vector<std::vector<long>> index(P);

  for (int n=0; n<ii_cpu.size(0); n++) {
    const int j = jj_data[n];
    const int k = kk_data[n];

    if (j >= t0 && j <= t1) {
      const int t = j - t0;
      graph[t].push_back(k);
      index[t].push_back(n);
    }
  }

  std::vector<long> ii_list, jj_list, idx, jdx;

  for (int i=0; i<P; i++) {
    for (int j=0; j<P; j++) {
      for (int k=0; k < graph[i].size(); k++) {
        for (int l=0; l < graph[j].size(); l++) {
          if (graph[i][k] == graph[j][l]) {
            ii_list.push_back(i);
            jj_list.push_back(j);

            idx.push_back(index[i][k]);
            idx.push_back(index[j][l]);
            idx.push_back(graph[i][k]);
          }
        }
      }
    }
  }

  torch::Tensor ix_cuda = torch::from_blob(idx.data(), {long(idx.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).to(torch::kCUDA).view({-1, 3});

  torch::Tensor jx_cuda = torch::stack({kk_cpu}, -1)
    .to(torch::kCUDA).to(torch::kInt64);

  torch::Tensor ii2_cpu = torch::from_blob(ii_list.data(), {long(ii_list.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).view({-1});

  torch::Tensor jj2_cpu = torch::from_blob(jj_list.data(), {long(jj_list.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).view({-1});

  torch::Tensor S = torch::zeros({ix_cuda.size(0), 6, 6}, 
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  torch::Tensor v = torch::zeros({jx_cuda.size(0), 6},
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));


  // parallel computation of E Q E^T
  EEt6x6_kernel<<<ix_cuda.size(0), THREADS>>>(
    E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    ix_cuda.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
    S.packed_accessor32<float,3,torch::RestrictPtrTraits>());

  // parallel computation of E Q w
  Ev6x1_kernel<<<jx_cuda.size(0), THREADS>>>(
    E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    w.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    jx_cuda.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
    v.packed_accessor32<float,2,torch::RestrictPtrTraits>());

  // schur block
  SparseBlockAsym A(P * 6 + n_intr);
  int M = 6;

  std::vector<T> extTripletList;

  // 4x4 lower right square of E Q E^T
  torch::Tensor CDCDT = torch::zeros({kx.size(0), n_intr, n_intr}, torch::TensorOptions()
          .device(torch::kCUDA).dtype(torch::kFloat32));

  CDCDT_kernel<<<kx.size(0), THREADS>>>(
    CD.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    CDCDT.packed_accessor32<float,3,torch::RestrictPtrTraits>()
  );
  torch::Tensor CDCDT_flat = torch::sum(CDCDT, 0).to(torch::kCUDA).to(torch::kFloat32);

  for (int i=0; i<n_intr; i++){
    for (int j=0; j<n_intr; j++){
      double val = CDCDT_flat[i][j].item<double>();
      extTripletList.push_back(T(6*P+i, 6*P+j, val));
    }
  }

  torch::Tensor ECDT_ext = torch::zeros({jx_cuda.size(0), 6, n_intr}, torch::TensorOptions()
         .device(torch::kCUDA).dtype(torch::kFloat32));

  ECDT_kernel<<<jx_cuda.size(0), THREADS>>>(
    E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    CD.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    jx_cuda.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
    ECDT_ext.packed_accessor32<float,3,torch::RestrictPtrTraits>()
  );


  torch::Tensor ECDT = torch::zeros({6*P, n_intr}, torch::TensorOptions()
         .device(torch::kCUDA).dtype(torch::kFloat32));

  for (int n=0; n<n_intr; n++){
    for (int i=0; i<jx_cuda.size(0); i++){
      int ki = jj_cpu[i].item<int>() - t0; 
      if (ki >= 0){
        for (int l=0; l<6; l++){
          ECDT[ki*6+l][n] += ECDT_ext[i][l][n];
        }
      }
    }
  }
  

  for (int j=0; j<n_intr; j++){
    for (int i=0; i<6*P; i++){
      double val = ECDT[i][j].item<double>();
      extTripletList.push_back(T(i, 6*P+j, val));
      extTripletList.push_back(T(6*P+j, i, val));
    }
  }

  torch::Tensor v_ext = torch::zeros({kx.size(0), n_intr}, torch::TensorOptions()
          .device(torch::kCUDA).dtype(torch::kFloat32));

  CDv_kernel<<<kx.size(0), THREADS>>>(
    CD.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    w.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    v_ext.packed_accessor32<float,2,torch::RestrictPtrTraits>());
  
  torch::Tensor v_ext_flat = torch::sum(v_ext, 0).to(torch::kCUDA).to(torch::kFloat32);

  A.update_lhs_extended(S, ii2_cpu, jj2_cpu, M, extTripletList);
  A.update_rhs_extended(v, jj_cpu - t0, M, v_ext_flat);

  return A;

}


SparseBlock schur_block(torch::Tensor E,
                        torch::Tensor Q,
                        torch::Tensor w,
                        torch::Tensor ii,
                        torch::Tensor jj,
                        torch::Tensor kk,
                        const int t0,
                        const int t1)
{

  torch::Tensor ii_cpu = ii.to(torch::kCPU);
  torch::Tensor jj_cpu = jj.to(torch::kCPU);
  torch::Tensor kk_cpu = kk.to(torch::kCPU);

  const int P = t1 - t0;
  const long* ii_data = ii_cpu.data_ptr<long>();
  const long* jj_data = jj_cpu.data_ptr<long>();
  const long* kk_data = kk_cpu.data_ptr<long>();

  std::vector<std::vector<long>> graph(P);
  std::vector<std::vector<long>> index(P);

  for (int n=0; n<ii_cpu.size(0); n++) {
    const int j = jj_data[n];
    const int k = kk_data[n];

    if (j >= t0 && j <= t1) {
      const int t = j - t0;
      graph[t].push_back(k);
      index[t].push_back(n);
    }
  }

  std::vector<long> ii_list, jj_list, idx, jdx;

  for (int i=0; i<P; i++) {
    for (int j=0; j<P; j++) {
      for (int k=0; k < graph[i].size(); k++) {
        for (int l=0; l < graph[j].size(); l++) {
          if (graph[i][k] == graph[j][l]) {
            ii_list.push_back(i);
            jj_list.push_back(j);

            idx.push_back(index[i][k]);
            idx.push_back(index[j][l]);
            idx.push_back(graph[i][k]);
          }
        }
      }
    }
  }

  torch::Tensor ix_cuda = torch::from_blob(idx.data(), {long(idx.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).to(torch::kCUDA).view({-1, 3});

  torch::Tensor jx_cuda = torch::stack({kk_cpu}, -1)
    .to(torch::kCUDA).to(torch::kInt64);

  torch::Tensor ii2_cpu = torch::from_blob(ii_list.data(), {long(ii_list.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).view({-1});

  torch::Tensor jj2_cpu = torch::from_blob(jj_list.data(), {long(jj_list.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).view({-1});

  torch::Tensor S = torch::zeros({ix_cuda.size(0), 6, 6}, 
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  torch::Tensor v = torch::zeros({jx_cuda.size(0), 6},
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));


  EEt6x6_kernel<<<ix_cuda.size(0), THREADS>>>(
    E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    ix_cuda.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
    S.packed_accessor32<float,3,torch::RestrictPtrTraits>());

  Ev6x1_kernel<<<jx_cuda.size(0), THREADS>>>(
    E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    w.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    jx_cuda.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
    v.packed_accessor32<float,2,torch::RestrictPtrTraits>());

  SparseBlock A(P, 6);

  A.update_lhs(S, ii2_cpu, jj2_cpu);
  A.update_rhs(v, jj_cpu - t0);

  return A;
}



std::vector<torch::Tensor> ba_cuda(
    torch::Tensor poses, 
    torch::Tensor disps, 
    torch::Tensor intrinsics, 
    torch::Tensor disps_sens, 
    torch::Tensor targets, 
    torch::Tensor weights, 
    torch::Tensor eta,
    torch::Tensor ii, 
    torch::Tensor jj, 
    const int t0,
    const int t1,
    const int iterations,
    const int model_id,
    const float lm,
    const float ep,
    const bool motion_only,
    const bool opt_intr,
    const float far_thres)
{
  auto opts = poses.options();
  const int num = ii.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);
  const int n_intr = (model_id == 2) ? 1 : intrinsics.size(0);

  torch::Tensor ts = torch::arange(t0, t1).to(torch::kCUDA);
  torch::Tensor ii_exp = torch::cat({ts, ii}, 0);
  torch::Tensor jj_exp = torch::cat({ts, jj}, 0);

  std::tuple<torch::Tensor, torch::Tensor> kuniq = 
    torch::_unique(ii_exp, true, true);

  torch::Tensor kx = std::get<0>(kuniq); 
  torch::Tensor kk_exp = std::get<1>(kuniq); 
    
  torch::Tensor dx;
  torch::Tensor dz;
  torch::Tensor dI;
  torch::Tensor dxdI;

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num, 6, 6}, opts);
  torch::Tensor vs = torch::zeros({2, num, 6}, opts);
  torch::Tensor Eii = torch::zeros({num, 6, ht*wd}, opts);
  torch::Tensor Eij = torch::zeros({num, 6, ht*wd}, opts);
  torch::Tensor Cii = torch::zeros({num, ht*wd}, opts);
  torch::Tensor wi = torch::zeros({num, ht*wd}, opts);

  torch::Tensor Calib = torch::zeros({num, n_intr, n_intr}, opts);
  torch::Tensor CalibPose = torch::zeros({num, 2, n_intr, 6}, opts);
  torch::Tensor CalibDepth = torch::zeros({num, ht*wd, n_intr}, opts);
  torch::Tensor q = torch::zeros({num, n_intr}, opts);

  for (int itr=0; itr<iterations; itr++) {

    projective_transform_kernel<<<num, THREADS>>>(
      targets.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      weights.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
      ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      vs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Eii.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Eij.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cii.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      wi.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      Calib.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      CalibPose.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      CalibDepth.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      opt_intr,
      n_intr,
      model_id
      );

    SparseBlock A(t1 - t0, 6);

    A.update_lhs(Hs.reshape({-1, 6, 6}), 
        torch::cat({ii, ii, jj, jj}) - t0, 
        torch::cat({ii, jj, ii, jj}) - t0);

    A.update_rhs(vs.reshape({-1, 6}), 
        torch::cat({ii, jj}) - t0);

    if (motion_only) {
      dx = A.solve(lm, ep);

      // update poses
      pose_retr_kernel<<<1, THREADS>>>(
        poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(), t0, t1);
    }

    else if (opt_intr) {
      // add depth residual if there are depth sensor measurements
      const float alpha = 0.05;
      torch::Tensor m = (disps_sens.index({kx, "..."}) > 0).to(torch::TensorOptions().dtype(torch::kFloat32)).view({-1, ht*wd}); 
      // far threshold
      m = m * (disps_sens.index({kx, "..."}) > 1.0/far_thres).to(torch::TensorOptions().dtype(torch::kFloat32)).view({-1, ht*wd});
      m = m * (disps.index({kx, "..."}) > 1.0/far_thres).to(torch::TensorOptions().dtype(torch::kFloat32)).view({-1, ht*wd});

      torch::Tensor residual = (disps.index({kx, "..."}) - disps_sens.index({kx, "..."})).view({-1, ht*wd});
      torch::Tensor C = accum_cuda(Cii, ii, kx) + m * alpha + (1 - m) * eta.view({-1, ht*wd});
      torch::Tensor w = accum_cuda(wi, ii, kx) - m * alpha * residual; 

      torch::Tensor Q = 1.0 / C;

      torch::Tensor Ei = accum_cuda(Eii.view({num, 6*ht*wd}), ii, ts).view({t1-t0, 6, ht*wd});
      torch::Tensor E = torch::cat({Ei, Eij}, 0);

      torch::Tensor CD_block = accum_cuda(CalibDepth.view({num, n_intr*ht*wd}), ii, kx).view(\
                               {-1, ht*wd, n_intr}).permute({0, 2, 1});

      SparseBlockAsym A = pose_calib_block(Hs, vs, Calib, q, CalibPose, ii, jj, t0, t1);
      SparseBlockAsym S = schur_calib_block(E, Q, CD_block, w, ii_exp, jj_exp, kk_exp, kx, t0, t1);

      dxdI = (A - S).solve(lm, ep);

      dx = torch::zeros({(t1 - t0)*6}, torch::TensorOptions()
          .device(torch::kCUDA).dtype(torch::kFloat32));

      dI = torch::zeros({n_intr}, torch::TensorOptions()
          .device(torch::kCUDA).dtype(torch::kFloat32));
      
      for (int i=0; i<(t1 - t0)*6; i++){
        dx[i] = dxdI[i];
      }

      dx = dx.view({(t1 - t0), 6});

      for (int i=0; i<n_intr; i++){
        dI[i] = dxdI[(t1 - t0)*6+i];
      }
      
      torch::Tensor ix = jj_exp - t0;
      torch::Tensor dw = torch::zeros({ix.size(0), ht*wd}, opts);

      EvT6x1_kernel<<<ix.size(0), THREADS>>>(
        E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        ix.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        dw.packed_accessor32<float,2,torch::RestrictPtrTraits>());
      
      torch::Tensor dw_add = torch::zeros({kx.size(0), ht*wd}, opts);

      CDvT_kernel<<<kx.size(0), THREADS>>>(
        CD_block.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dI.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        dw_add.packed_accessor32<float,2,torch::RestrictPtrTraits>());
      

      dz = Q * (w - accum_cuda(dw, ii_exp, kx) + dw_add); 

      // update poses
      pose_retr_kernel<<<1, THREADS>>>(
        poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(), t0, t1);

      // update disparity maps
      disp_retr_kernel<<<kx.size(0), THREADS>>>(
        disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dz.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        kx.packed_accessor32<long,1,torch::RestrictPtrTraits>());
      

      // for the "focal" model we must update fx and fy, but we scale the update with their ratio
      if (model_id == 2){
        const float fx = intrinsics[0].item<float>();;
        const float fy = intrinsics[1].item<float>();;
        const float focal_ratio = fy / fx;
        intrinsics[1] += focal_ratio * dI[0];
      }

      // update intrinsics
      for (int i=0; i<n_intr; i++){
        intrinsics[i] += dI[i];
      }
      

      
    }
    
    else {
      // add depth residual if there are depth sensor measurements
      const float alpha = 0.05;
      torch::Tensor m = (disps_sens.index({kx, "..."}) > 0).to(torch::TensorOptions().dtype(torch::kFloat32)).view({-1, ht*wd}); 
      // far threshold
      m = m * (disps_sens.index({kx, "..."}) > 1.0/far_thres).to(torch::TensorOptions().dtype(torch::kFloat32)).view({-1, ht*wd});
      m = m * (disps.index({kx, "..."}) > 1.0/far_thres).to(torch::TensorOptions().dtype(torch::kFloat32)).view({-1, ht*wd});

      torch::Tensor residual = (disps.index({kx, "..."}) - disps_sens.index({kx, "..."})).view({-1, ht*wd});
      torch::Tensor C = accum_cuda(Cii, ii, kx) + m * alpha + (1 - m) * eta.view({-1, ht*wd});
      torch::Tensor w = accum_cuda(wi, ii, kx) - m * alpha * residual;
      torch::Tensor Q = 1.0 / C;

      torch::Tensor Ei = accum_cuda(Eii.view({num, 6*ht*wd}), ii, ts).view({t1-t0, 6, ht*wd});
      torch::Tensor E = torch::cat({Ei, Eij}, 0);

      SparseBlock S = schur_block(E, Q, w, ii_exp, jj_exp, kk_exp, t0, t1);
      dx = (A - S).solve(lm, ep);

      torch::Tensor ix = jj_exp - t0;
      torch::Tensor dw = torch::zeros({ix.size(0), ht*wd}, opts);

      EvT6x1_kernel<<<ix.size(0), THREADS>>>(
        E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        ix.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        dw.packed_accessor32<float,2,torch::RestrictPtrTraits>());

      dz = Q * (w - accum_cuda(dw, ii_exp, kx));

      // update poses
      pose_retr_kernel<<<1, THREADS>>>(
        poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(), t0, t1);

      // update disparity maps
      disp_retr_kernel<<<kx.size(0), THREADS>>>(
        disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dz.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        kx.packed_accessor32<long,1,torch::RestrictPtrTraits>());
    }

  }

  return {dx, dz, dI};
}



torch::Tensor frame_distance_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj,
    const float beta,
    const int model_id)
{
  auto opts = poses.options();
  const int num = ii.size(0);

  torch::Tensor dist = torch::zeros({num}, opts);

  frame_distance_kernel<<<num, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    dist.packed_accessor32<float,1,torch::RestrictPtrTraits>(), beta, model_id);

  return dist;
}


std::vector<torch::Tensor> projmap_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj)
{
  auto opts = poses.options();
  const int num = ii.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  torch::Tensor coords = torch::zeros({num, ht, wd, 3}, opts);
  torch::Tensor valid = torch::zeros({num, ht, wd, 1}, opts);

  projmap_kernel<<<num, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    valid.packed_accessor32<float,4,torch::RestrictPtrTraits>());

  return {coords, valid};
}


torch::Tensor depth_filter_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ix,
    torch::Tensor thresh,
    const int model_id)
{
  const int num = ix.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  torch::Tensor counter = torch::zeros({num, ht, wd}, disps.options());

  dim3 blocks(num, 6, NUM_BLOCKS(ht * wd));

  depth_filter_kernel<<<blocks, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ix.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    thresh.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    model_id,
    counter.packed_accessor32<float,3,torch::RestrictPtrTraits>());

  return counter;
}


torch::Tensor iproj_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    const int model_id)
{

  const int nm = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  auto opts = disps.options();
  torch::Tensor points = torch::zeros({nm, ht, wd, 3}, opts);

  dim3 blocks(nm, NUM_BLOCKS(ht * wd));

  iproj_kernel<<<blocks, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    model_id,
    points.packed_accessor32<float,4,torch::RestrictPtrTraits>());

  return points;

}
