#include <iostream>
#include <ATen/ATen.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F 
#endif

#define EPSILON_ABS_ZERO 1e-10
#define EPSILON_DIV_ZERO 1e-4

// for the older gpus atomicAdd with double arguments does not exist
#if  __CUDA_ARCH__ < 600 and defined(__CUDA_ARCH__)
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

namespace{

/* 
    Above-triangle test 
    Method: consider the tetrahedron constructed from the triangle and the query point and
    check whether the signed volume of the tetrahedron is positive
*/
template<typename scalar_t>
__device__ bool above_triangle_test(
    const scalar_t *v0, const scalar_t *v1, const scalar_t *v2, const scalar_t *p) {
    
    const scalar_t x1 = v1[0] - v0[0], y1 = v1[1] - v0[1], z1 = v1[2] - v0[2];
    const scalar_t x2 = v2[0] - v0[0], y2 = v2[1] - v0[1], z2 = v2[2] - v0[2];
    const scalar_t x3 =  p[0] - v0[0], y3 =  p[1] - v0[1], z3 =  p[2] - v0[2];
    return (x1*y2*z3 - x1*y3*z2 - x2*y1*z3 + x2*y3*z1 + x3*y1*z2 - x3*y2*z1) >= 0;
}

/* 
    In-tetrahedron test
    Method: check whether the query point is "above" the four triangle of the tetrahedron 
*/
template<typename scalar_t>
__device__ bool in_tetrahedron_test(const scalar_t *tet, const scalar_t* p) {
    bool flags[4];
    const int tris[3*4] {
        /* root, edge1, edge2 */
        0, 2, 1,
        0, 3, 2, 
        0, 1, 3, 
        1, 2, 3
    };
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const scalar_t* v0 = tet + 3 * tris[3*k+0];
        const scalar_t* v1 = tet + 3 * tris[3*k+1];
        const scalar_t* v2 = tet + 3 * tris[3*k+2];
        flags[k] = above_triangle_test(v0, v1, v2, p);
    }
    return flags[0] == flags[1] && flags[0] == flags[2] && flags[0] == flags[3];
}


/* 
    Voxel labeling
*/
__device__ void label_occupied_voxel(float *voxel) {
    atomicMax((int*)voxel, __float_as_int(1.0f));
}

__device__ void label_occupied_voxel(double *voxel) {
    atomicCAS((unsigned long long*)voxel, __double_as_longlong(0), __double_as_longlong(1.0));
}

/*
    Distance Calculator
*/
__device__ float calc_squared_dist(const float *p1, const float *p2) {
    const float x = p1[0] - p2[0];
    const float y = p1[1] - p2[1];
    const float z = p1[2] - p2[2];
    return x*x + y*y + z*z;
}

__device__ double calc_squared_dist(const double *p1, const double *p2) {
    const double x = p1[0] - p2[0];
    const double y = p1[1] - p2[1];
    const double z = p1[2] - p2[2];
    return x*x + y*y + z*z;
}
// __device__ float calc_dist(const float *p1, const float *p2) {
//     return __fsqrt_rn(calc_squared_dist(p1, p2));
// }

// __device__ double calc_dist(const double *p1, const double *p2) {
//     return __dsqrt_rn(calc_squared_dist(p1, p2));
// }


template<typename scalar_t>
__global__ void forward_voxelize_cuda_kernel(
    const scalar_t* __restrict__ tetrahedrons, 
    scalar_t* __restrict__ out_volume, 
    int batch_size, 
    int num_tets,  
    int volume_res) {
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * num_tets) {
        return;
    }
    const int vr = volume_res;
    const int bi = i / num_tets;
    const int ti = i % num_tets;
    const scalar_t voxel_size = 1.0 / volume_res;
    const int vr_half = vr / 2;
    const scalar_t* tet = &tetrahedrons[i * 12];
    scalar_t xmin = tet[0], ymin = tet[1], zmin = tet[2];
    scalar_t xmax = tet[0], ymax = tet[1], zmax = tet[2];
    #pragma unroll
    for (int k = 1; k < 4; k++) {
        xmin = fminf(xmin, tet[3*k + 0]);
        xmax = fmaxf(xmax, tet[3*k + 0]);
        ymin = fminf(ymin, tet[3*k + 1]);
        ymax = fmaxf(ymax, tet[3*k + 1]);
        zmin = fminf(zmin, tet[3*k + 2]);
        zmax = fmaxf(zmax, tet[3*k + 2]);
    }

    // checks the voxels in the bounding box of that tetrahedron
    const int xvmin = max(int(xmin/voxel_size + vr_half), 0);
    const int xvmax = min(int(xmax/voxel_size + vr_half) + 1, vr-1);
    const int yvmin = max(int(ymin/voxel_size + vr_half), 0);
    const int yvmax = min(int(ymax/voxel_size + vr_half) + 1, vr-1);
    const int zvmin = max(int(zmin/voxel_size + vr_half), 0);
    const int zvmax = min(int(zmax/voxel_size + vr_half) + 1, vr-1);
    for (int zz = zvmin; zz <= zvmax; zz++) 
        for (int yy = yvmin; yy <= yvmax; yy++)
            for (int xx = xvmin; xx <= xvmax; xx++) {
                #if 0
                const int dx[8] = {0, 1, 0, 1, 0, 1, 0, 1};
                const int dy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
                const int dz[8] = {0, 0, 0, 0, 1, 1, 1, 1};

                // if at least one corner of the voxel is inside the tetrahedron, 
                // then we consider the voxel is inside the tetrahedron
                #pragma unroll
                for (int k = 0; k < 8; k++) 
                {
                    const scalar_t px = (xx + dx[k] - vr_half) * voxel_size;
                    const scalar_t py = (yy + dy[k] - vr_half) * voxel_size;
                    const scalar_t pz = (zz + dz[k] - vr_half) * voxel_size;
                    const scalar_t pt[3] = {px, py, pz};
                    if (in_tetrahedron_test(tet, pt)) {
                        const int i_ = bi*vr*vr*vr + zz*vr*vr + yy*vr + xx;
                        label_occupied_voxel(&(out_volume[i_]));
                        break;
                    }
                }
                #else
                const scalar_t px = (xx + 0.5 - vr_half) * voxel_size;
                const scalar_t py = (yy + 0.5 - vr_half) * voxel_size;
                const scalar_t pz = (zz + 0.5 - vr_half) * voxel_size;
                const scalar_t pt[3] = {px, py, pz};
                if (in_tetrahedron_test(tet, pt)) {
                    const int i_ = bi*vr*vr*vr + zz*vr*vr + yy*vr + xx;
                    label_occupied_voxel(&(out_volume[i_]));
                }
            #endif
            }
}



template<typename scalar_t>
__global__ void forward_calc_semantic_volume_cuda_kernel(
    const scalar_t* __restrict__ occ_volume, 
    const scalar_t* __restrict__ smpl_vertices, 
    const scalar_t* __restrict__ smpl_vertex_code, 
    scalar_t* __restrict__ semantic_volume, 
    scalar_t* __restrict__ weight_sum_volume, 
    float sigma, 
    int batch_size, 
    int num_vertex, 
    int volume_res) {
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * volume_res * volume_res * volume_res) {
        return;
    }

    if (occ_volume[i] < 1e-3) { // empty voxel
        return;
    }

    const int vr = volume_res;
    const int vn = num_vertex;
    const int vr_half = vr / 2;
    const scalar_t voxel_size = 1.0 / volume_res;

    const int bi = i / (vr*vr*vr);
    const int vi = i % (vr*vr*vr);
    const int xv = vi % vr;
    const int yv = (vi/vr) % vr;
    const int zv = vi / (vr*vr);
    const scalar_t px = (xv + 0.5 - vr_half) * voxel_size;
    const scalar_t py = (yv + 0.5 - vr_half) * voxel_size;
    const scalar_t pz = (zv + 0.5 - vr_half) * voxel_size;
    const scalar_t pt[3] = {px, py, pz};

    const scalar_t* sv = smpl_vertices + bi * vn * 3;
    const scalar_t* sc = smpl_vertex_code + bi * vn * 3;

    scalar_t weight_sum = 1e-10;
    scalar_t code[3] = {(scalar_t)0};
    for (int k = 0; k < vn; k++) {
        const scalar_t d = calc_squared_dist(pt, sv + k*3);
        const scalar_t w = __expf(-d/(sigma*sigma));
        code[0] += w * sc[k * 3 + 0];
        code[1] += w * sc[k * 3 + 1];
        code[2] += w * sc[k * 3 + 2];
        weight_sum += w;
    }

    semantic_volume[3 * i + 0] = code[0] / weight_sum;
    semantic_volume[3 * i + 1] = code[1] / weight_sum;
    semantic_volume[3 * i + 2] = code[2] / weight_sum;
    weight_sum_volume[i] = weight_sum;
}

}


std::vector<at::Tensor> forward_semantic_voxelization_cuda(
    at::Tensor smpl_vertices, 
    at::Tensor smpl_vertex_code, 
    at::Tensor smpl_tetrahedrons, 
    at::Tensor occ_volume, 
    at::Tensor semantic_volume, 
    at::Tensor weight_sum_volume, 
    float sigma) {
    
    const auto batch_size = smpl_vertices.size(0);
    const auto num_vertex = smpl_vertices.size(1);
    const auto num_tets = smpl_tetrahedrons.size(1);
    const auto volume_res = occ_volume.size(1);
    
    const int threads = 512;
    const dim3 blocks_1 ((batch_size * num_tets - 1) / threads +1);

    AT_DISPATCH_FLOATING_TYPES(smpl_vertices.scalar_type(), "forward_voxelize_cuda_kernel", ([&] {
        forward_voxelize_cuda_kernel<scalar_t><<<blocks_1, threads>>>(
            smpl_tetrahedrons.data_ptr<scalar_t>(),
            occ_volume.data_ptr<scalar_t>(),
            batch_size, 
            num_tets, 
            volume_res);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in forward_voxelize_cuda_kernel: %s\n", cudaGetErrorString(err));

    const dim3 blocks_2 ((batch_size * volume_res * volume_res * volume_res - 1) / threads +1);
    AT_DISPATCH_FLOATING_TYPES(smpl_vertices.scalar_type(), "forward_calc_semantic_volume_cuda_kernel", ([&] {
        forward_calc_semantic_volume_cuda_kernel<scalar_t><<<blocks_2, threads>>>(
            occ_volume.data_ptr<scalar_t>(),
            smpl_vertices.data_ptr<scalar_t>(),
            smpl_vertex_code.data_ptr<scalar_t>(),
            semantic_volume.data_ptr<scalar_t>(),
            weight_sum_volume.data_ptr<scalar_t>(),
            sigma, 
            batch_size, 
            num_vertex, 
            volume_res);
    }));

    err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in forward_calc_semantic_volume_cuda_kernel: %s\n", cudaGetErrorString(err));

    return {occ_volume, semantic_volume, weight_sum_volume};
}

