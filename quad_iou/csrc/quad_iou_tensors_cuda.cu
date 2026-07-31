/******************************************************************************
 * Copyright [2024] Irakli Salia
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#define QUAD_ELEMENTS 8
#define MAX_ALL_POINTS 16
#define THREAD_COUNT_X 16
#define THREAD_COUNT_Y 16
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <cmath>
#include "polygonArea.h"
#include "insidePoints.h"
#include "intersectionPoints.h"
#include "sortPoints.h"
#include "allPoints.h"
#include "simpleIntersectCheck.h"
#include "checks.h"

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(...)            \
    THO_DISPATCH_CASE(torch::headeronly::ScalarType::Double, __VA_ARGS__) \
    THO_DISPATCH_CASE(torch::headeronly::ScalarType::Float, __VA_ARGS__)  \
    THO_DISPATCH_CASE(torch::headeronly::ScalarType::Half, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...) \
    THO_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(__VA_ARGS__))


template <typename scalar_t>
__device__ inline scalar_t intersectionArea(
    const scalar_t quad_0[QUAD_ELEMENTS],
    const scalar_t quad_1[QUAD_ELEMENTS]
) {
    // If we know that quad_0 and quad_1 are not
    // intersecting even a tiny bit(minimum enclosing box check)
    // we can skip below calculation altogether
    if (!simpleIntersectCheck::checkSimpleIntersection(quad_0, quad_1)) return 0.0;

    scalar_t all_points[MAX_ALL_POINTS][2];
    allPoints::fillPointsWithInfinity(all_points);

    // saving numIntersections so inside_points can be written
    // after those points
    int numIntersections = intersectionPoints::findIntersectionPoints(quad_0,
                                                                      quad_1,
                                                                      all_points);
    insidePoints::findPointsInside(quad_0, quad_1, all_points, numIntersections);
    sortPoints::sortPointsClockwise(all_points);
    scalar_t intersectArea = polygonArea::calcPolygonArea(all_points);
    return intersectArea;
}

template <typename scalar_t>
__device__ inline scalar_t unionArea(int quad_0_idx,
                                     int quad_1_idx,
                                     int quad_0_size,
                                     scalar_t *polygonAreas,
                                     scalar_t intersectArea) {
    return polygonAreas[quad_0_idx] + \
                polygonAreas[quad_0_size + quad_1_idx] - \
                    intersectArea;
}

template <typename scalar_t>
__device__ inline scalar_t calculateIoU(
    const scalar_t quad_0[QUAD_ELEMENTS],
    const scalar_t quad_1[QUAD_ELEMENTS],
    int quad_0_idx,
    int quad_1_idx,
    int quad_0_size,
    scalar_t *polygonAreas) {

    const scalar_t epsilon = std::numeric_limits<scalar_t>::epsilon();

    scalar_t intersect_area = intersectionArea(quad_0, quad_1);
    return intersect_area / (unionArea(quad_0_idx, quad_1_idx, quad_0_size, polygonAreas, intersect_area) + epsilon);
}

template <typename scalar_t>
__global__ void calculateIoUKernel(
    scalar_t *quad_0,
    scalar_t *quad_1,
    scalar_t *iou_matrix,
    scalar_t *polygonAreas,
    int quad_0_size,
    int quad_1_size
    ) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Shared memory for storing quads
    __shared__ scalar_t quad_0_shared[THREAD_COUNT_X][QUAD_ELEMENTS];
    __shared__ scalar_t quad_1_shared[THREAD_COUNT_Y][QUAD_ELEMENTS];
    if (tx < QUAD_ELEMENTS && idx2 < quad_1_size) {
        quad_1_shared[ty][tx] = quad_1[idx2 * QUAD_ELEMENTS + tx];
    }
    if (ty < QUAD_ELEMENTS && idx1 < quad_0_size) {
        quad_0_shared[tx][ty] = quad_0[idx1 * QUAD_ELEMENTS + ty];
    }
    __syncthreads();

    if ((idx1 < quad_0_size) && (idx2 < quad_1_size)) {
        iou_matrix[idx1 * quad_1_size + idx2] = calculateIoU(quad_0_shared[tx],
                                                             quad_1_shared[ty],
                                                             idx1,
                                                             idx2,
                                                             quad_0_size,
                                                             polygonAreas);
    }
}


template <typename scalar_t>
__global__ void polygonAreaCalculationKernel(
    scalar_t *polygonAreas,
    scalar_t *quad_0,
    scalar_t *quad_1,
    int quad_0_size,
    int quad_1_size,
    bool sort_input_quads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < quad_0_size) {
        scalar_t *quadrilateral = &quad_0[idx * 4 * 2];
        if (sort_input_quads){
            sortPoints::sortQuadPointsClockwise(quadrilateral);    
        }
        polygonAreas[idx] = polygonArea::calcQuadrilateralArea(quadrilateral);
    } else if (idx < (quad_0_size + quad_1_size)) {
        scalar_t *quadrilateral = &quad_1[(idx - quad_0_size) * 4 * 2];
        if (sort_input_quads){
            sortPoints::sortQuadPointsClockwise(quadrilateral);    
        }
        polygonAreas[idx] = polygonArea::calcQuadrilateralArea(quadrilateral);
    }
}

namespace quad_iou {

torch::stable::Tensor calculate_iou_cuda(torch::stable::Tensor quad_0, torch::stable::Tensor quad_1, bool sort_input_quads) {
    checks::check_tensor_validity(quad_0, quad_1);
    STD_TORCH_CHECK(quad_0.device().type() == torch::headeronly::DeviceType::CUDA, "quad_0 must be a CUDA tensor");
    STD_TORCH_CHECK(quad_1.device().type() == torch::headeronly::DeviceType::CUDA, "quad_1 must be a CUDA tensor");

    // Create an output tensor
    torch::stable::Tensor iou_matrix = torch::stable::empty({quad_0.size(0), quad_1.size(0)}, quad_0.scalar_type(), quad_0.layout(), quad_0.device());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(quad_0.scalar_type(), "calculate_iou_cuda", ([&] {        
        // Allocate device memory for polygon areas
        scalar_t* polygonAreas_d;
        cudaMalloc((void**)&polygonAreas_d, (quad_0.size(0) + quad_1.size(0)) * sizeof(scalar_t));

        // Define block and grid sizes
        dim3 blockSizeQuad(128, 1, 1);
        dim3 gridSizeQuad((quad_0.size(0) + quad_1.size(0) + blockSizeQuad.x - 1) / blockSizeQuad.x, 1, 1);
        dim3 blockSize(THREAD_COUNT_X, THREAD_COUNT_Y);
        dim3 gridSize((quad_0.size(0) + blockSize.x - 1) / blockSize.x, 
                      (quad_1.size(0) + blockSize.y - 1) / blockSize.y);

        // If sorting of input quads is needed
        if (sort_input_quads) {
            torch::stable::Tensor quad_0_copy = torch::stable::clone(quad_0);
            torch::stable::Tensor quad_1_copy = torch::stable::clone(quad_1);
            // Calculate polygon areas for sorted quads
            polygonAreaCalculationKernel<scalar_t><<<gridSizeQuad, blockSizeQuad>>>(
                polygonAreas_d,
                quad_0_copy.mutable_data_ptr<scalar_t>(),
                quad_1_copy.mutable_data_ptr<scalar_t>(),
                quad_0.size(0),
                quad_1.size(0),
                sort_input_quads
            );
            cudaDeviceSynchronize();

            // Calculate IoU for sorted quads
            calculateIoUKernel<scalar_t><<<gridSize, blockSize>>>(
                quad_0_copy.mutable_data_ptr<scalar_t>(),
                quad_1_copy.mutable_data_ptr<scalar_t>(),
                iou_matrix.mutable_data_ptr<scalar_t>(),
                polygonAreas_d,
                quad_0.size(0),
                quad_1.size(0)
            );
        } else {
            // Calculate polygon areas for unsorted quads
            polygonAreaCalculationKernel<scalar_t><<<gridSizeQuad, blockSizeQuad>>>(
                polygonAreas_d,
                quad_0.mutable_data_ptr<scalar_t>(),
                quad_1.mutable_data_ptr<scalar_t>(),
                quad_0.size(0),
                quad_1.size(0),
                sort_input_quads
            );
            cudaDeviceSynchronize();

            // Calculate IoU for unsorted quads
            calculateIoUKernel<scalar_t><<<gridSize, blockSize>>>(
                quad_0.mutable_data_ptr<scalar_t>(),
                quad_1.mutable_data_ptr<scalar_t>(),
                iou_matrix.mutable_data_ptr<scalar_t>(),
                polygonAreas_d,
                quad_0.size(0),
                quad_1.size(0)
            );
        }

        // Synchronize and free device memory
        cudaDeviceSynchronize();
        cudaFree(polygonAreas_d);
    }));
    return iou_matrix;
}

// Registers CUDA implementations
STABLE_TORCH_LIBRARY_IMPL(quad_iou, CUDA, m) {
  m.impl("calculate_iou", TORCH_BOX(&calculate_iou_cuda));
}

}