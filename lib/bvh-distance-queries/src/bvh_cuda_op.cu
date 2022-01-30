/*
 * Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
 * holder of all proprietary rights on this computer program.
 * You can only use this computer program if you have closed
 * a license agreement with MPG or you get the right to use the computer
 * program from someone who is authorized to grant you that right.
 * Any use of the computer program without a valid license is prohibited and
 * liable to prosecution.
 *
 * Copyright©2019 Max-Planck-Gesellschaft zur Förderung
 * der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
 * for Intelligent Systems. All rights reserved.
 *
 * @author Vasileios Choutas
 * Contact: vassilis.choutas@tuebingen.mpg.de
 * Contact: ps-license@tuebingen.mpg.de
 *
 */

#include <torch/extension.h>
#include <torch/types.h>

#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "aabb.hpp"
#include "defs.hpp"
#include "double_vec_ops.h"
#include "helper_math.h"
#include "math_utils.hpp"
#include "priority_queue.hpp"
#include "triangle.hpp"

// Number of threads per block for CUDA kernel launch
#ifndef NUM_THREADS
#define NUM_THREADS 256
#endif

#ifndef FORCE_INLINE
#define FORCE_INLINE 1
#endif /* ifndef FORCE_INLINE */

#ifndef PROFILING
#define PROFILING 0
#endif /* ifndef PROFILING */

#ifndef ERROR_CHECKING
#define ERROR_CHECKING 1
#endif /* ifndef ERROR_CHECKING */

// Macro for checking cuda errors following a cuda launch or api call
#if ERROR_CHECKING == 1
#define cudaCheckError()                                                       \
  {                                                                            \
    cudaDeviceSynchronize();                                                   \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(e));                                           \
      exit(0);                                                                 \
    }                                                                          \
  }
#else
#define cudaCheckError()
#endif

typedef unsigned int MortonCode;

template <typename T>
std::ostream &operator<<(std::ostream &os, const vec3<T> &x) {
  os << x.x << ", " << x.y << ", " << x.z;
  return os;
}

std::ostream &operator<<(std::ostream &os, const vec3<float> &x) {
  os << x.x << ", " << x.y << ", " << x.z;
  return os;
}

std::ostream &operator<<(std::ostream &os, const vec3<double> &x) {
  os << x.x << ", " << x.y << ", " << x.z;
  return os;
}

template <typename T> std::ostream &operator<<(std::ostream &os, vec3<T> x) {
  os << x.x << ", " << x.y << ", " << x.z;
  return os;
}

__host__ __device__ inline double3 fmin(const double3 &a, const double3 &b) {
  return make_double3(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));
}

__host__ __device__ inline double3 fmax(const double3 &a, const double3 &b) {
  return make_double3(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));
}

struct is_valid_cnt : public thrust::unary_function<long2, int> {
public:
  __host__ __device__ int operator()(long2 vec) const {
    return vec.x >= 0 && vec.y >= 0;
  }
};

template <typename T>
__host__ __device__ T pointToTriangleDistance(vec3<T> p,
                                              TrianglePtr<T> tri_ptr) {
  vec3<T> a = tri_ptr->v0;
  vec3<T> b = tri_ptr->v1;
  vec3<T> c = tri_ptr->v2;

  vec3<T> ba = b - a;
  vec3<T> pa = p - a;
  vec3<T> cb = c - b;
  vec3<T> pb = p - b;
  vec3<T> ac = a - c;
  vec3<T> pc = p - c;
  vec3<T> nor = cross(ba, ac);

  return (sign<T>(dot(cross(ba, nor), pa)) + sign<T>(dot(cross(cb, nor), pb)) +
              sign<T>(dot(cross(ac, nor), pc)) <
          2.0)
             ? min(min(dot2<T>(ba * clamp(dot(ba, pa) / dot2<T>(ba), 0.0, 1.0) -
                               pa),
                       dot2<T>(cb * clamp(dot(cb, pb) / dot2<T>(cb), 0.0, 1.0) -
                               pb)),
                   dot2<T>(ac * clamp(dot(ac, pc) / dot2<T>(ac), 0.0, 1.0) -
                           pc))
             : dot(nor, pa) * dot(nor, pa) / dot2<T>(nor);
}

template <typename T>
__host__ __device__ T pointToTriangleDistance(vec3<T> p, TrianglePtr<T> tri_ptr,
                                              vec3<T> *closest_bc,
                                              vec3<T> *closest_point) {
  vec3<T> a = tri_ptr->v0;
  vec3<T> b = tri_ptr->v1;
  vec3<T> c = tri_ptr->v2;

  // Check if P in vertex region outside A
  vec3<T> ab = b - a;
  vec3<T> ac = c - a;
  vec3<T> ap = p - a;
  T d1 = dot(ab, ap);
  T d2 = dot(ac, ap);
  if (d1 <= static_cast<T>(0) && d2 <= static_cast<T>(0)) {
    *closest_point = a;
    *closest_bc = make_vec3<T>(static_cast<T>(1.0), static_cast<T>(0.0), static_cast<T>(0.0));
    return dot(ap, ap);
  }
  // Check if P in vertex region outside B
  vec3<T> bp = p - b;
  T d3 = dot(ab, bp);
  T d4 = dot(ac, bp);

  if (d3 >= 0.0f && d4 <= d3) {
    *closest_point = b;
    *closest_bc = make_vec3<T>(static_cast<T>(0.0), static_cast<T>(1.0), static_cast<T>(0.0));
    return dot(bp, bp);
  }
  // Check if P in edge region of AB, if so return projection of P onto AB
  T vc = d1 * d4 - d3 * d2;
  if (vc <= static_cast<T>(0) && d1 >= static_cast<T>(0) &&
      d3 <= static_cast<T>(0)) {
    T v = d1 / (d1 - d3);
    *closest_point = a + v * ab;
    *closest_bc = make_vec3<T>(static_cast<T>(1 - v), static_cast<T>(v), static_cast<T>(0.0));
    return dot(p - *closest_point, p - *closest_point);
  }
  // Check if P in vertex region outside C
  vec3<T> cp = p - c;
  T d5 = dot(ab, cp);
  T d6 = dot(ac, cp);
  if (d6 >= static_cast<T>(0) && d5 <= d6) {
    *closest_point = c;
    *closest_bc = make_vec3<T>(0.0, 0.0, 1.0);
    return dot(cp, cp);
  }
  // Check if P in edge region of AC, if so return projection of P onto AC
  T vb = d5 * d2 - d1 * d6;
  if (vb <= static_cast<T>(0) && d2 >= static_cast<T>(0) &&
      d6 <= static_cast<T>(0)) {
    T w = d2 / (d2 - d6);
    *closest_point = a + w * ac;
    *closest_bc = make_vec3<T>(static_cast<T>(1 - w), static_cast<T>(0.0), static_cast<T>(w));
    return dot(p - *closest_point, p - *closest_point);
  }
  // Check if P in edge region of BC, if so return projection of P onto BC
  T va = d3 * d6 - d5 * d4;
  if (va <= static_cast<T>(0) && (d4 - d3) >= static_cast<T>(0) &&
      (d5 - d6) >= static_cast<T>(0)) {
    T w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    *closest_point = b + w * (c - b);
    *closest_bc = make_vec3<T>(static_cast<T>(0), static_cast<T>(1 - w), static_cast<T>(w));
    return dot(p - *closest_point, p - *closest_point);
  }
  // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
  T denom = static_cast<T>(1) / (va + vb + vc);
  T v = vb * denom;
  T w = vc * denom;
  *closest_point = a + v * ab + w * ac;
  *closest_bc = make_vec3<T>(static_cast<T>(1 - v - w), static_cast<T>(v), static_cast<T>(w));
  return dot(p - *closest_point, p - *closest_point);
}

template <typename T>
__global__ void ComputeTriBoundingBoxes(Triangle<T> *triangles,
                                        int num_triangles, AABB<T> *bboxes) {
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < num_triangles;
       idx += blockDim.x * gridDim.x) {
    bboxes[idx] = triangles[idx].ComputeBBox();
  }
  return;
}

template <typename T> struct BVHNode {
public:
  AABB<T> bbox;

  // __host__ __device__
  // BVHNode(): left(nullptr), right(nullptr), tri_ptr(nullptr), idx(-1);

  TrianglePtr<T> tri_ptr;
  BVHNode<T> *left;
  BVHNode<T> *right;
  BVHNode<T> *parent;
  __host__ __device__ inline bool isLeaf() { return !left && !right; };

  // The index of the object contained in the node
  int idx;
};

template <typename T> using BVHNodePtr = BVHNode<T> *;

template <typename T>
__device__
#if FORCE_INLINE == 1
    __forceinline__
#endif
    bool
    checkOverlap(const AABB<T> &bbox1, const AABB<T> &bbox2) {
  return (bbox1.min_t.x <= bbox2.max_t.x) && (bbox1.max_t.x >= bbox2.min_t.x) &&
         (bbox1.min_t.y <= bbox2.max_t.y) && (bbox1.max_t.y >= bbox2.min_t.y) &&
         (bbox1.min_t.z <= bbox2.max_t.z) && (bbox1.max_t.z >= bbox2.min_t.z);
}

template <typename T, int StackSize = 32>
__device__ T traverseBVHStack(const vec3<T> &queryPoint, BVHNodePtr<T> root,
                              long *closest_face, vec3<T> *closest_bc,
                              vec3<T> *closestPoint) {
  BVHNodePtr<T> stack[StackSize];
  BVHNodePtr<T> *stackPtr = stack;
  *stackPtr++ = nullptr; // push

  BVHNodePtr<T> node = root;
  T closest_distance = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;

  do {
    // Check each child node for overlap.
    BVHNodePtr<T> childL = node->left;
    BVHNodePtr<T> childR = node->right;

    T distance_left = pointToAABBDistance<T>(queryPoint, childL->bbox);
    T distance_right = pointToAABBDistance<T>(queryPoint, childR->bbox);

    bool checkL = distance_left <= closest_distance;
    bool checkR = distance_right <= closest_distance;

    if (checkL && childL->isLeaf()) {
      // If  the child is a leaf then
      TrianglePtr<T> tri_ptr = childL->tri_ptr;
      vec3<T> curr_clos_point;
      vec3<T> curr_closest_bc;

      T distance_left = pointToTriangleDistance<T>(
          queryPoint, tri_ptr, &curr_closest_bc, &curr_clos_point);
      if (distance_left <= closest_distance) {
        closest_distance = distance_left;
        *closest_face = childL->idx;
        *closestPoint = curr_clos_point;
        *closest_bc = curr_closest_bc;
      }
    }

    if (checkR && childR->isLeaf()) {
      // If  the child is a leaf then
      TrianglePtr<T> tri_ptr = childR->tri_ptr;
      vec3<T> curr_clos_point;
      vec3<T> curr_closest_bc;

      T distance_right = pointToTriangleDistance<T>(
          queryPoint, tri_ptr, &curr_closest_bc, &curr_clos_point);
      if (distance_right <= closest_distance) {
        closest_distance = distance_right;
        *closest_face = childR->idx;
        *closestPoint = curr_clos_point;
        *closest_bc = curr_closest_bc;
      }
    }
    // Query overlaps an internal node => traverse.
    bool traverseL = (checkL && !childL->isLeaf());
    bool traverseR = (checkR && !childR->isLeaf());

    if (!traverseL && !traverseR) {
      node = *--stackPtr; // pop
    } else {
      node = (traverseL) ? childL : childR;
      if (traverseL && traverseR) {
        *stackPtr++ = childR; // push
      }
    }
  } while (node != nullptr);

  return closest_distance;
}

template <typename T, int QueueSize = 32>
__device__ T traverseBVH(const vec3<T> &queryPoint, BVHNodePtr<T> root,
                         long *closest_face, vec3<T> *closest_bc,
                         vec3<T> *closestPoint) {
  // Create a priority queue
  PriorityQueue<T, BVHNodePtr<T>, QueueSize> queue;

  T root_dist = pointToAABBDistance(queryPoint, root->bbox);

  queue.insert_key(root_dist, root);

  BVHNodePtr<T> node = nullptr;

  T closest_distance = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;

  while (queue.get_size() > 0) {
    std::pair<T, BVHNodePtr<T>> output = queue.extract();
    // T curr_distance = output.first;
    node = output.second;

    // Check each child node for overlap.
    BVHNodePtr<T> childL = node->left;
    BVHNodePtr<T> childR = node->right;

    T distance_left = pointToAABBDistance<T>(queryPoint, childL->bbox);
    T distance_right = pointToAABBDistance<T>(queryPoint, childR->bbox);

    if (distance_left <= closest_distance) {
      if (childL->isLeaf()) {
        // If  the child is a leaf then
        TrianglePtr<T> tri_ptr = childL->tri_ptr;
        vec3<T> curr_clos_point;
        vec3<T> curr_closest_bc;

        T distance_left = pointToTriangleDistance<T>(
            queryPoint, tri_ptr, &curr_closest_bc, &curr_clos_point);
        if (distance_left <= closest_distance) {
          closest_distance = distance_left;
          *closest_face = childL->idx;
          *closestPoint = curr_clos_point;
          *closest_bc = curr_closest_bc;
        }
      } else {
        queue.insert_key(distance_left, childL);
      }
    }

    if (distance_right <= closest_distance) {
      if (childR->isLeaf()) {
        // If  the child is a leaf then
        TrianglePtr<T> tri_ptr = childR->tri_ptr;
        vec3<T> curr_clos_point;
        vec3<T> curr_closest_bc;

        T distance_right = pointToTriangleDistance<T>(
            queryPoint, tri_ptr, &curr_closest_bc, &curr_clos_point);
        if (distance_right <= closest_distance) {
          closest_distance = distance_right;
          *closest_face = childR->idx;
          *closestPoint = curr_clos_point;
          *closest_bc = curr_closest_bc;
        }
      } else {
        queue.insert_key(distance_right, childR);
      }
    }
  }

  return closest_distance;
}

template <typename T, int QueueSize = 32>
__global__ void findNearestNeighbor(vec3<T> *query_points, T *distances,
                                    vec3<T> *closest_points,
                                    long *closest_faces,
                                    vec3<T> *closest_bcs,
                                    BVHNodePtr<T> root, int num_points,
                                    bool use_stack = true) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_points;
       idx += blockDim.x * gridDim.x) {
    vec3<T> query_point = query_points[idx];

    long closest_face;
    vec3<T> closest_bc;
    vec3<T> closest_point;

    T closest_distance;
    if (use_stack) {
      closest_distance = traverseBVHStack<T, QueueSize>(
          query_point, root, &closest_face, &closest_bc, &closest_point);
    } else {
      closest_distance = traverseBVH<T, QueueSize>(
          query_point, root, &closest_face, &closest_bc, &closest_point);
    }
    distances[idx] = closest_distance;
    closest_points[idx] = closest_point;
    closest_faces[idx] = closest_face;
    closest_bcs[idx] = closest_bc;
  }
  return;
}

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__
#if FORCE_INLINE == 1
    __forceinline__
#endif
        MortonCode
        expandBits(MortonCode v) {
  // Shift 16
  v = (v * 0x00010001u) & 0xFF0000FFu;
  // Shift 8
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  // Shift 4
  v = (v * 0x00000011u) & 0xC30C30C3u;
  // Shift 2
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
template <typename T>
__device__
#if FORCE_INLINE == 1
    __forceinline__
#endif
        MortonCode
        morton3D(T x, T y, T z) {
  x = min(max(x * 1024.0f, 0.0f), 1023.0f);
  y = min(max(y * 1024.0f, 0.0f), 1023.0f);
  z = min(max(z * 1024.0f, 0.0f), 1023.0f);
  MortonCode xx = expandBits((MortonCode)x);
  MortonCode yy = expandBits((MortonCode)y);
  MortonCode zz = expandBits((MortonCode)z);
  return xx * 4 + yy * 2 + zz;
}

template <typename T>
__global__ void ComputePointMortonCodes(vec3<T> *points, vec3<T> *in_points,
                                        int num_points,
                                        MortonCode *morton_codes) {
  AABB<T> scene_bb(-1, -1, -1, 1, 1, 1);
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < num_points;
       idx += blockDim.x * gridDim.x) {
    // Fetch the current triangle
    vec3<T> point = in_points[idx];

    T x = (point.x - scene_bb.min_t.x) / (scene_bb.max_t.x - scene_bb.min_t.x);
    T y = (point.y - scene_bb.min_t.y) / (scene_bb.max_t.y - scene_bb.min_t.y);
    T z = (point.z - scene_bb.min_t.z) / (scene_bb.max_t.z - scene_bb.min_t.z);

    morton_codes[idx] = morton3D<T>(x, y, z);
    points[idx] = point;
  }
  return;
}

template <typename T>
__global__ void ComputeMortonCodes(Triangle<T> *triangles, int num_triangles,
                                   AABB<T> *scene_bb,
                                   MortonCode *morton_codes) {
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < num_triangles;
       idx += blockDim.x * gridDim.x) {
    // Fetch the current triangle
    Triangle<T> tri = triangles[idx];
    vec3<T> centroid = (tri.v0 + tri.v1 + tri.v2) / (T)3.0;

    T x = (centroid.x - scene_bb->min_t.x) /
          (scene_bb->max_t.x - scene_bb->min_t.x);
    T y = (centroid.y - scene_bb->min_t.y) /
          (scene_bb->max_t.y - scene_bb->min_t.y);
    T z = (centroid.z - scene_bb->min_t.z) /
          (scene_bb->max_t.z - scene_bb->min_t.z);

    morton_codes[idx] = morton3D<T>(x, y, z);
  }
  return;
}

__device__
#if FORCE_INLINE == 1
    __forceinline__
#endif
    int
    LongestCommonPrefix(int i, int j, MortonCode *morton_codes,
                        int num_triangles, int *triangle_ids) {
  // This function will be called for i - 1, i, i + 1, so we might go beyond
  // the array limits
  if (i < 0 || i > num_triangles - 1 || j < 0 || j > num_triangles - 1)
    return -1;

  MortonCode key1 = morton_codes[i];
  MortonCode key2 = morton_codes[j];

  if (key1 == key2) {
    // Duplicate key:__clzll(key1 ^ key2) will be equal to the number of
    // bits in key[1, 2]. Add the number of leading zeros between the
    // indices
    return __clz(key1 ^ key2) + __clz(triangle_ids[i] ^ triangle_ids[j]);
  } else {
    // Keys are different
    return __clz(key1 ^ key2);
  }
}

template <typename T>
__global__ void BuildRadixTree(MortonCode *morton_codes, int num_triangles,
                               int *triangle_ids, BVHNodePtr<T> internal_nodes,
                               BVHNodePtr<T> leaf_nodes) {
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < num_triangles - 1;
       idx += blockDim.x * gridDim.x) {
    // if (idx >= num_triangles - 1)
    // return;

    int delta_next = LongestCommonPrefix(idx, idx + 1, morton_codes,
                                         num_triangles, triangle_ids);
    int delta_last = LongestCommonPrefix(idx, idx - 1, morton_codes,
                                         num_triangles, triangle_ids);
    // Find the direction of the range
    int direction = delta_next - delta_last >= 0 ? 1 : -1;

    int delta_min = LongestCommonPrefix(idx, idx - direction, morton_codes,
                                        num_triangles, triangle_ids);

    // Do binary search to compute the upper bound for the length of the range
    int lmax = 2;
    while (LongestCommonPrefix(idx, idx + lmax * direction, morton_codes,
                               num_triangles, triangle_ids) > delta_min) {
      lmax *= 2;
    }

    // Use binary search to find the other end.
    int l = 0;
    int divider = 2;
    for (int t = lmax / divider; t >= 1; divider *= 2) {
      if (LongestCommonPrefix(idx, idx + (l + t) * direction, morton_codes,
                              num_triangles, triangle_ids) > delta_min) {
        l = l + t;
      }
      t = lmax / divider;
    }
    int j = idx + l * direction;

    // Find the length of the longest common prefix for the current node
    int node_delta =
        LongestCommonPrefix(idx, j, morton_codes, num_triangles, triangle_ids);
    int s = 0;
    divider = 2;
    // Search for the split position using binary search.
    for (int t = (l + (divider - 1)) / divider; t >= 1; divider *= 2) {
      if (LongestCommonPrefix(idx, idx + (s + t) * direction, morton_codes,
                              num_triangles, triangle_ids) > node_delta) {
        s = s + t;
      }
      t = (l + (divider - 1)) / divider;
    }
    // gamma in the Karras paper
    int split = idx + s * direction + min(direction, 0);

    // Assign the parent and the left, right children for the current node
    BVHNodePtr<T> curr_node = internal_nodes + idx;
    if (min(idx, j) == split) {
      curr_node->left = leaf_nodes + split;
      (leaf_nodes + split)->parent = curr_node;
    } else {
      curr_node->left = internal_nodes + split;
      (internal_nodes + split)->parent = curr_node;
    }
    if (max(idx, j) == split + 1) {
      curr_node->right = leaf_nodes + split + 1;
      (leaf_nodes + split + 1)->parent = curr_node;
    } else {
      curr_node->right = internal_nodes + split + 1;
      (internal_nodes + split + 1)->parent = curr_node;
    }
  }
  return;
}

template <typename T>
__global__ void CreateHierarchy(BVHNodePtr<T> internal_nodes,
                                BVHNodePtr<T> leaf_nodes, int num_triangles,
                                Triangle<T> *triangles, int *triangle_ids,
                                int *atomic_counters) {
  // int idx = blockDim.x * blockIdx.x + threadIdx.x;
  // if (idx >= num_triangles)
  // return;
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < num_triangles;
       idx += blockDim.x * gridDim.x) {

    BVHNodePtr<T> leaf = leaf_nodes + idx;
    // Assign the index to the primitive
    leaf->idx = triangle_ids[idx];

    Triangle<T> tri = triangles[triangle_ids[idx]];
    // Assign the bounding box of the triangle to the leaves
    leaf->bbox = tri.ComputeBBox();
    leaf->tri_ptr = &triangles[triangle_ids[idx]];
    // leaf->tri_ptr = &triangles[idx];

    BVHNodePtr<T> curr_node = leaf->parent;
    int current_idx = curr_node - internal_nodes;

    // Increment the atomic counter
    int curr_counter = atomicAdd(atomic_counters + current_idx, 1);
    while (true) {
      // atomicAdd returns the old value at the specified address. Thus the
      // first thread to reach this point will immediately return
      if (curr_counter == 0)
        break;

      // Calculate the bounding box of the current node as the union of the
      // bounding boxes of its children.
      AABB<T> left_bb = curr_node->left->bbox;
      AABB<T> right_bb = curr_node->right->bbox;
      curr_node->bbox = left_bb + right_bb;
      // If we have reached the root break
      if (curr_node == internal_nodes)
        break;

      // Proceed to the parent of the node
      curr_node = curr_node->parent;
      // Calculate its position in the flat array
      current_idx = curr_node - internal_nodes;
      // Update the visitation counter
      curr_counter = atomicAdd(atomic_counters + current_idx, 1);
    }
  }

  return;
}

template <typename T>
__global__ void copy_to_tensor(T *dest, T *source, int *ids, int num_elements) {
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < num_elements;
       idx += blockDim.x * gridDim.x) {
    // dest[idx] = source[ids[idx]];
    dest[ids[idx]] = source[idx];
  }
  return;
}

template <typename T, int blockSize = NUM_THREADS>
void buildBVH(BVHNodePtr<T> internal_nodes, BVHNodePtr<T> leaf_nodes,
              Triangle<T> *__restrict__ triangles,
              thrust::device_vector<int> *triangle_ids, int num_triangles,
              int batch_size) {

#if PRINT_TIMINGS == 1
  // Create the CUDA events used to estimate the execution time of each
  // kernel.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
#endif

  thrust::device_vector<AABB<T>> bounding_boxes(num_triangles);

  int gridSize = (num_triangles + blockSize - 1) / blockSize;
#if PRINT_TIMINGS == 1
  cudaEventRecord(start);
#endif
  // Compute the bounding box for all the triangles
#if DEBUG_PRINT == 1
  std::cout << "Start computing triangle bounding boxes" << std::endl;
#endif
  ComputeTriBoundingBoxes<T><<<gridSize, blockSize>>>(
      triangles, num_triangles, bounding_boxes.data().get());
#if PRINT_TIMINGS == 1
  cudaEventRecord(stop);
#endif

  cudaCheckError();

#if DEBUG_PRINT == 1
  std::cout << "Finished computing triangle bounding_boxes" << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Compute Triangle Bounding boxes = " << milliseconds << " (ms)"
            << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventRecord(start);
#endif
  // Compute the union of all the bounding boxes
  AABB<T> host_scene_bb = thrust::reduce(
      bounding_boxes.begin(), bounding_boxes.end(), AABB<T>(), MergeAABB<T>());
#if PRINT_TIMINGS == 1
  cudaEventRecord(stop);
#endif

  cudaCheckError();

#if DEBUG_PRINT == 1
  std::cout << "Finished Calculating scene Bounding Box" << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Scene bounding box reduction = " << milliseconds << " (ms)"
            << std::endl;
#endif

  // TODO: Custom reduction ?
  // Copy the bounding box back to the GPU
  AABB<T> *scene_bb_ptr;
  cudaMalloc(&scene_bb_ptr, sizeof(AABB<T>));
  cudaMemcpy(scene_bb_ptr, &host_scene_bb, sizeof(AABB<T>),
             cudaMemcpyHostToDevice);

  thrust::device_vector<MortonCode> morton_codes(num_triangles);
#if DEBUG_PRINT == 1
  std::cout << "Start Morton Code calculation ..." << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventRecord(start);
#endif
  // Compute the morton codes for the centroids of all the primitives
  ComputeMortonCodes<T><<<gridSize, blockSize>>>(
      triangles, num_triangles, scene_bb_ptr, morton_codes.data().get());
#if PRINT_TIMINGS == 1
  cudaEventRecord(stop);
#endif

  cudaCheckError();

#if DEBUG_PRINT == 1
  std::cout << "Finished calculating Morton Codes ..." << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Morton code calculation = " << milliseconds << " (ms)"
            << std::endl;
#endif

#if DEBUG_PRINT == 1
  std::cout << "Creating triangle ID sequence" << std::endl;
#endif
  // Construct an array of triangle ids.
  thrust::sequence(triangle_ids->begin(), triangle_ids->end());
#if DEBUG_PRINT == 1
  std::cout << "Finished creating triangle ID sequence ..." << std::endl;
#endif

  // Sort the triangles according to the morton code
#if DEBUG_PRINT == 1
  std::cout << "Starting Morton Code sorting!" << std::endl;
#endif

  try {
#if PRINT_TIMINGS == 1
    cudaEventRecord(start);
#endif
    thrust::sort_by_key(morton_codes.begin(), morton_codes.end(),
                        triangle_ids->begin());
#if PRINT_TIMINGS == 1
    cudaEventRecord(stop);
#endif
#if DEBUG_PRINT == 1
    std::cout << "Finished morton code sorting!" << std::endl;
#endif
#if PRINT_TIMINGS == 1
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Morton code sorting = " << milliseconds << " (ms)"
              << std::endl;
#endif
  } catch (thrust::system_error e) {
    std::cout << "Error inside Morton code sort: " << e.what() << std::endl;
  }

#if DEBUG_PRINT == 1
  std::cout << "Start building radix tree" << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventRecord(start);
#endif
  // Construct the radix tree using the sorted morton code sequence
  BuildRadixTree<T><<<gridSize, blockSize>>>(
      morton_codes.data().get(), num_triangles, triangle_ids->data().get(),
      internal_nodes, leaf_nodes);
#if PRINT_TIMINGS == 1
  cudaEventRecord(stop);
#endif

  cudaCheckError();

#if DEBUG_PRINT == 1
  std::cout << "Finished radix tree" << std::endl;
#endif
#if PRINT_TIMINGS == 1
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Building radix tree = " << milliseconds << " (ms)" << std::endl;
#endif
  // Create an array that contains the atomic counters for each node in the
  // tree
  thrust::device_vector<int> counters(num_triangles);

#if DEBUG_PRINT == 1
  std::cout << "Start Linear BVH generation" << std::endl;
#endif
  // Build the Bounding Volume Hierarchy in parallel from the leaves to the
  // root
  CreateHierarchy<T><<<gridSize, blockSize>>>(
      internal_nodes, leaf_nodes, num_triangles, triangles,
      triangle_ids->data().get(), counters.data().get());

  cudaCheckError();

#if PRINT_TIMINGS == 1
  cudaEventRecord(stop);
#endif
#if DEBUG_PRINT == 1
  std::cout << "Finished with LBVH generation ..." << std::endl;
#endif

#if PRINT_TIMINGS == 1
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Hierarchy generation = " << milliseconds << " (ms)"
            << std::endl;
#endif

  cudaFree(scene_bb_ptr);
  return;
}


void bvh_distance_queries_kernel(
    const torch::Tensor &triangles, const torch::Tensor &points,
    torch::Tensor *distances, torch::Tensor *closest_points,
    torch::Tensor *closest_faces, torch::Tensor *closest_bcs,
    int queue_size = 128, bool sort_points_by_morton = true) {

  const auto batch_size = triangles.size(0);
  const auto num_triangles = triangles.size(1);
  const auto num_points = points.size(1);

  thrust::device_vector<int> triangle_ids(num_triangles);

  int blockSize = NUM_THREADS;

  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
  int gridSize = std::min(
      32 * numSMs, static_cast<int>((num_points + blockSize - 1) / blockSize));

  // Construct the bvh tree
  AT_DISPATCH_FLOATING_TYPES(
      triangles.type(), "bvh_tree_building", ([&] {
      // using scalar_t = float;

#if PRINT_TIMINGS == 1
        // Create the CUDA events used to estimate the execution time of each
        // kernel.
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
#endif

        scalar_t *distances_ptr;
        cudaMalloc((void **)&distances_ptr, num_points * sizeof(scalar_t));
        cudaCheckError();

        vec3<scalar_t> *morton_sorted_points_ptr;
        cudaMalloc((void **)&morton_sorted_points_ptr,
                   num_points * sizeof(vec3<scalar_t>));
        cudaCheckError();

        vec3<scalar_t> *closest_points_ptr;
        cudaMalloc((void **)&closest_points_ptr,
                   num_points * sizeof(vec3<scalar_t>));
        cudaCheckError();

        long *closest_faces_ptr;
        cudaMalloc((void **)&closest_faces_ptr, num_points * sizeof(long));
        cudaCheckError();

        vec3<scalar_t> *closest_bcs_ptr;
        cudaMalloc((void **)&closest_bcs_ptr, num_points * sizeof(vec3<scalar_t>));
        cudaCheckError();

        // The thrust vectors that contain the BVH nodes
        thrust::device_vector<BVHNode<scalar_t>> leaf_nodes(num_triangles);
        thrust::device_vector<BVHNode<scalar_t>> internal_nodes(num_triangles -
                                                                1);

        auto triangle_scalar_t_ptr = triangles.data<scalar_t>();

        for (int bidx = 0; bidx < batch_size; ++bidx) {

          Triangle<scalar_t> *triangles_ptr =
              (TrianglePtr<scalar_t>)triangle_scalar_t_ptr +
              num_triangles * bidx;

#if DEBUG_PRINT == 1
          std::cout << "Start building BVH" << std::endl;
#endif
          buildBVH<scalar_t, NUM_THREADS>(
              internal_nodes.data().get(), leaf_nodes.data().get(),
              triangles_ptr, &triangle_ids, num_triangles, batch_size);
#if DEBUG_PRINT == 1
          std::cout << "Successfully built BVH" << std::endl;
#endif
          cudaCheckError();

#if DEBUG_PRINT == 1
          std::cout << "Start BVH traversal" << std::endl;
#endif
          vec3<scalar_t> *points_ptr =
              (vec3<scalar_t> *)points.data<scalar_t>() + num_points * bidx;
          thrust::device_vector<int> point_ids(num_points);
          thrust::sequence(point_ids.begin(), point_ids.end());

          if (sort_points_by_morton) {
            thrust::device_vector<MortonCode> morton_codes(num_points);

#if PRINT_TIMINGS == 1
            cudaEventRecord(start);
#endif
            ComputePointMortonCodes<scalar_t><<<gridSize, NUM_THREADS>>>(
                // morton_sorted_points.data().get(), points_ptr, num_points,
                morton_sorted_points_ptr, points_ptr, num_points,
                morton_codes.data().get());
            cudaCheckError();
            cudaDeviceSynchronize();
#if PRINT_TIMINGS == 1
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "Compute morton codes for input points = "
                      << milliseconds << " (ms)" << std::endl;
#endif

            thrust::device_ptr<vec3<scalar_t>> dev_ptr =
                thrust::device_pointer_cast(morton_sorted_points_ptr);

            thrust::sort_by_key(morton_codes.begin(), morton_codes.end(),
                                thrust::make_zip_iterator(thrust::make_tuple(
                                    point_ids.begin(), dev_ptr)));
            cudaCheckError();

            points_ptr = morton_sorted_points_ptr;
          }

#ifdef PROFILING
          cudaProfilerStart();
#endif
          if (queue_size == 32) {
            findNearestNeighbor<scalar_t, 32><<<gridSize, NUM_THREADS>>>(
                points_ptr, distances_ptr, closest_points_ptr,
                closest_faces_ptr, closest_bcs_ptr,
                internal_nodes.data().get(), num_points);
          } else if (queue_size == 64) {
            findNearestNeighbor<scalar_t, 64><<<gridSize, NUM_THREADS>>>(
                points_ptr, distances_ptr, closest_points_ptr,
                closest_faces_ptr, closest_bcs_ptr,
                internal_nodes.data().get(), num_points);
          } else if (queue_size == 128) {
            findNearestNeighbor<scalar_t, 128><<<gridSize, NUM_THREADS>>>(
                points_ptr, distances_ptr, closest_points_ptr,
                closest_faces_ptr, closest_bcs_ptr,
                internal_nodes.data().get(), num_points);
          } else if (queue_size == 256) {
            findNearestNeighbor<scalar_t, 256><<<gridSize, NUM_THREADS>>>(
                points_ptr, distances_ptr, closest_points_ptr,
                closest_faces_ptr, closest_bcs_ptr,
                internal_nodes.data().get(), num_points);
          } else if (queue_size == 512) {
            findNearestNeighbor<scalar_t, 512><<<gridSize, NUM_THREADS>>>(
                points_ptr, distances_ptr, closest_points_ptr,
                closest_faces_ptr, closest_bcs_ptr,
                internal_nodes.data().get(), num_points);
          } else if (queue_size == 1024) {
            findNearestNeighbor<scalar_t, 1024><<<gridSize, NUM_THREADS>>>(
                points_ptr, distances_ptr, closest_points_ptr,
                closest_faces_ptr, closest_bcs_ptr,
                internal_nodes.data().get(), num_points);
          }
          cudaCheckError();
#ifdef PROFILING
          cudaProfilerStop();
#endif

          scalar_t *distances_dest_ptr =
              (scalar_t *)distances->data<scalar_t>() + num_points * bidx;
          vec3<scalar_t> *closest_points_dest_ptr =
              (vec3<scalar_t> *)closest_points->data<scalar_t>() +
              num_points * bidx;
          vec3<scalar_t> *closest_bcs_dest_ptr =
              (vec3<scalar_t> *)closest_bcs->data<scalar_t>() + num_points * bidx;
          long *closest_faces_dest_ptr =
              closest_faces->data<long>() + num_points * bidx;
          if (sort_points_by_morton) {
            copy_to_tensor<scalar_t>
                <<<gridSize, NUM_THREADS>>>(distances_dest_ptr, distances_ptr,
                                            point_ids.data().get(), num_points);
            copy_to_tensor<vec3<scalar_t>><<<gridSize, NUM_THREADS>>>(
                closest_points_dest_ptr, closest_points_ptr,
                point_ids.data().get(), num_points);
            copy_to_tensor<vec3<scalar_t> ><<<gridSize, NUM_THREADS>>>(
                closest_bcs_dest_ptr, closest_bcs_ptr,
                point_ids.data().get(), num_points);
            copy_to_tensor<long><<<gridSize, NUM_THREADS>>>(
                closest_faces_dest_ptr, closest_faces_ptr,
                point_ids.data().get(), num_points);
          } else {
            cudaMemcpy(distances_dest_ptr, distances_ptr,
                       num_points * sizeof(scalar_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(closest_points_dest_ptr, closest_points_ptr,
                       num_points * sizeof(vec3<scalar_t>),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(closest_bcs_dest_ptr, closest_bcs_ptr,
                       num_points * sizeof(vec3<scalar_t>), cudaMemcpyDeviceToDevice);
            cudaMemcpy(closest_faces_dest_ptr, closest_faces_ptr,
                       num_points * sizeof(long), cudaMemcpyDeviceToDevice);
          }

#if DEBUG_PRINT == 1
          std::cout << "Successfully finished BVH traversal" << std::endl;
#endif
        }
        cudaFree(distances_ptr);
        cudaFree(closest_points_ptr);
        cudaFree(closest_faces_ptr);
        cudaFree(closest_bcs_ptr);
        cudaFree(morton_sorted_points_ptr);
      }));
}
