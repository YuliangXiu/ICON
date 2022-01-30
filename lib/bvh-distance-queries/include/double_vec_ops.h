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


#ifndef DOUBLE_VEC_OPS_H
#define DOUBLE_VEC_OPS_H

#include "cuda_runtime.h"

inline __host__ __device__ double2 operator+(double2 a, double2 b) {
    return make_double2(a.x + b.x, a.y + b.y);
}


inline __host__ __device__ double3 operator+(double3 a, double3 b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ void operator/=(double2 &a, double2 b) {
    a.x /= b.x;
    a.y /= b.y;
}

inline __host__ __device__ double2 operator/(double2 a, double b) {
    return make_double2(a.x / b, a.y / b);
}

inline __host__ __device__ double3 operator/(double3 a, double3 b) {
    return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}


inline __host__ __device__ double3 operator*(double a, double3 b) {
    return make_double3(a * b.x, a * b.y, a * b.z);
}

inline __host__ __device__ double3 operator*(double3 a, double3 b) {
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ void operator/=(double3 &a, double3 b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}

inline __host__ __device__ double3 operator/(double3 a, double b) {
    return make_double3(a.x / b, a.y / b, a.z / b);
}

inline __host__ __device__ double dot(double2 a, double2 b) {
    return a.x * b.x + a.y * b.y;
}

inline __host__ __device__ double dot(double3 a, double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ double3 cross(double3 a, double3 b)
{
    return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

inline __host__ __device__ double2 operator-(double2 a, double2 b)
{
    return make_double2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(double2 &a, double2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ double2 operator-(double2 a, double b)
{
    return make_double2(a.x - b, a.y - b);
}
inline __host__ __device__ double2 operator-(double b, double2 a)
{
    return make_double2(b - a.x, b - a.y);
}

inline __host__ __device__ double3 operator-(double3 a, double3 b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(double3 &a, double3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ double3 operator-(double3 a, double b)
{
    return make_double3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ double3 operator-(double b, double3 a)
{
    return make_double3(b - a.x, b - a.y, b - a.z);
}

#endif // ifndef DOUBLE_VEC_OPS_H
