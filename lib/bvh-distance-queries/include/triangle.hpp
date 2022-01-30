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

#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "defs.hpp"
#include "double_vec_ops.h"
#include "helper_math.h"

#include "math_utils.hpp"
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

template <typename T>
__align__(48)
    struct Triangle {
public:
  vec3<T> v0;
  vec3<T> v1;
  vec3<T> v2;

  __host__ __device__ Triangle() {}
  __host__ __device__ Triangle(vec3<T> vertex0, vec3<T> vertex1,
                               vec3<T> vertex2)
      : v0(vertex0), v1(vertex1), v2(vertex2){};
  __host__ __device__ Triangle(const vec3<T> &vertex0, const vec3<T> &vertex1,
                               const vec3<T> &vertex2)
      : v0(vertex0), v1(vertex1), v2(vertex2){};

  __host__ __device__ AABB<T> ComputeBBox() {
    return AABB<T>(min(v0.x, min(v1.x, v2.x)), min(v0.y, min(v1.y, v2.y)),
                   min(v0.z, min(v1.z, v2.z)), max(v0.x, max(v1.x, v2.x)),
                   max(v0.y, max(v1.y, v2.y)), max(v0.z, max(v1.z, v2.z)));
  }
};

template <typename T> using TrianglePtr = Triangle<T> *;

template <typename T>
std::ostream &operator<<(std::ostream &os, const Triangle<T> &x) {
  os << x.v0 << std::endl;
  os << x.v1 << std::endl;
  os << x.v2 << std::endl;
  return os;
}


#endif // TRIANGLE_H
