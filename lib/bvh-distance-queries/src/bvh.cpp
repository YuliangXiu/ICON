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



#include <iostream>
#include <vector>
#include <limits>

#include <torch/extension.h>


void bvh_distance_queries_kernel(const torch::Tensor& triangles,
        const torch::Tensor& points,
        torch::Tensor* distances,
        torch::Tensor* closest_points,
        torch::Tensor* closest_faces,
        torch::Tensor* closest_bcs,
        int queue_size=128,
        bool sort_points_by_morton=true
        );

// void bvh_self_distance_queries_kernel(torch::Tensor triangles,
        // torch::Tensor* distances,
        // torch::Tensor* closest_points
        // );

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> bvh_distance_queries(torch::Tensor triangles,
        torch::Tensor points,
        int queue_size=128,
        bool sort_points_by_morton=true) {
    CHECK_INPUT(triangles);
    CHECK_INPUT(points);

    auto options = torch::TensorOptions()
        .dtype(triangles.dtype())
        .layout(triangles.layout())
        .device(triangles.device());

    torch::Tensor distances = torch::full({
            triangles.size(0), points.size(1)},
            -1, options);
    torch::Tensor closest_points = torch::full({
            triangles.size(0), points.size(1), 3},
            -1, options);
    torch::Tensor closest_bcs = torch::full({
            triangles.size(0), points.size(1), 3}, 0,
            options);
    torch::Tensor closest_faces = torch::full({
            triangles.size(0), points.size(1)},
            -1, torch::TensorOptions()
        .dtype(torch::kLong)
        .layout(triangles.layout())
        .device(triangles.device()));

    bvh_distance_queries_kernel(triangles,
            points, &distances, &closest_points, &closest_faces,
            &closest_bcs,
            queue_size, sort_points_by_morton);

    return {distances, closest_points, closest_faces, closest_bcs};
}

// std::vector<torch::Tensor> bvh_self_distance_queries(torch::Tensor triangles) {
    // CHECK_INPUT(triangles);

    // torch::Tensor distances = torch::full({
            // triangles.size(0), triangles.size(1)},
            // -1, torch::device(triangles.device()).dtype(triangles.dtype()));
    // torch::Tensor closest_points = torch::full({
            // triangles.size(0), triangles.size(1), 3},
            // -1, torch::device(triangles.device()).dtype(triangles.dtype()));

    // return {distances, closest_points};
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("distance_queries", &bvh_distance_queries, "BVH distance queries forward (CUDA)",
        py::arg("triangles"), py::arg("points"),
        py::arg("queue_size") = 128,
        py::arg("sort_points_by_morton") = true
        );
  // m.def("self_distance_queries", &bvh_self_distance_queries, "BVH self distance queries forward (CUDA)",
        // py::arg("triangles"));
}
