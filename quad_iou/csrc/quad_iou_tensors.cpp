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
#include <Python.h>

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

#define AT_DISPATCH_CASE_FLOATING_TYPES(...)            \
    THO_DISPATCH_CASE(torch::headeronly::ScalarType::Double, __VA_ARGS__) \
    THO_DISPATCH_CASE(torch::headeronly::ScalarType::Float, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
    THO_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))


template <typename scalar_t>
inline scalar_t intersectionArea(
    scalar_t *quad_0,
    scalar_t *quad_1
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
inline scalar_t unionArea(int quad_0_idx,
                          int quad_1_idx,
                          int quad_0_size,
                          scalar_t *polygonAreas,
                          scalar_t intersectArea) {
    return polygonAreas[quad_0_idx] + \
                polygonAreas[quad_0_size + quad_1_idx] - \
                    intersectArea;
}

template <typename scalar_t>
inline void calculateIoU(
    scalar_t *quad_0,
    scalar_t *quad_1,
    scalar_t *iou_matrix,
    scalar_t *polygonAreas,
    int quad_0_size,
    int quad_1_size
    ) {
    const scalar_t epsilon = std::numeric_limits<scalar_t>::epsilon();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < quad_0_size; i++){
        for (int j = 0; j < quad_1_size; j++){
            scalar_t *box_0 = &quad_0[i * 8];
            scalar_t *box_1 = &quad_1[j * 8];
            scalar_t intersect_area = intersectionArea(box_0, box_1);
            scalar_t union_area = unionArea(i, j, quad_0_size, polygonAreas, intersect_area) + epsilon;
            iou_matrix[i * quad_1_size + j] = intersect_area / union_area;
        }
    }
}


template <typename scalar_t>
void polygonAreaCalculation(
    scalar_t *polygonAreas,
    scalar_t *quad_0,
    scalar_t *quad_1,
    int quad_0_size,
    int quad_1_size,
    bool sort_input_quads
) {
    #pragma omp parallel for
    for (int i = 0; i < (quad_0_size + quad_1_size); i++) {
        scalar_t *box;
        if (i < quad_0_size) {
            box = &quad_0[i * 4 * 2];
        } else {
            box = &quad_1[(i - quad_0_size) * 4 * 2];
        }
        if (sort_input_quads) {
            sortPoints::sortQuadPointsClockwise(box);
        }
        polygonAreas[i] = polygonArea::calcQuadrilateralArea(box);
    }
}

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the STABLE_TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace quad_iou {

torch::stable::Tensor calculate_iou_cpu(torch::stable::Tensor quad_0, torch::stable::Tensor quad_1, bool sort_input_quads) {
    checks::check_tensor_validity(quad_0, quad_1);
    // Create an output tensor
    torch::stable::Tensor iou_matrix = torch::stable::empty({quad_0.size(0), quad_1.size(0)}, quad_0.scalar_type(), quad_0.layout(), quad_0.device());

    AT_DISPATCH_FLOATING_TYPES(quad_0.scalar_type(), "calculate_iou_cpu", ([&] {
        // Allocate device memory for polygon areas
        scalar_t* polygonAreas = (scalar_t*)malloc((quad_0.size(0) + quad_1.size(0)) * sizeof(scalar_t));
        // If sorting of input quads is needed
        if (sort_input_quads) {
            torch::stable::Tensor quad_0_copy = torch::stable::clone(quad_0);
            torch::stable::Tensor quad_1_copy = torch::stable::clone(quad_1);
            // Calculate polygon areas for sorted quads
            polygonAreaCalculation<scalar_t>(
                polygonAreas,
                quad_0_copy.mutable_data_ptr<scalar_t>(),
                quad_1_copy.mutable_data_ptr<scalar_t>(),
                quad_0.size(0),
                quad_1.size(0),
                sort_input_quads
            );

            // Calculate IoU for sorted quads
            calculateIoU<scalar_t>(
                quad_0_copy.mutable_data_ptr<scalar_t>(),
                quad_1_copy.mutable_data_ptr<scalar_t>(),
                iou_matrix.mutable_data_ptr<scalar_t>(),
                polygonAreas,
                quad_0.size(0),
                quad_1.size(0)
            );
        } else {
            // Calculate polygon areas for unsorted quads
            polygonAreaCalculation<scalar_t>(
                polygonAreas,
                quad_0.mutable_data_ptr<scalar_t>(),
                quad_1.mutable_data_ptr<scalar_t>(),
                quad_0.size(0),
                quad_1.size(0),
                sort_input_quads
            );
            // Calculate IoU for unsorted quads
            calculateIoU<scalar_t>(
                quad_0.mutable_data_ptr<scalar_t>(),
                quad_1.mutable_data_ptr<scalar_t>(),
                iou_matrix.mutable_data_ptr<scalar_t>(),
                polygonAreas,
                quad_0.size(0),
                quad_1.size(0)
            );
        }
    }));
    return iou_matrix;
}


STABLE_TORCH_LIBRARY(quad_iou, m) {
  m.def("calculate_iou(Tensor quad_0, Tensor quad_1, bool sort_input_quads) -> Tensor");
}

// Registers CPU implementations for mymuladd, mymul, myadd_out
STABLE_TORCH_LIBRARY_IMPL(quad_iou, CPU, m) {
  m.impl("calculate_iou", TORCH_BOX(&calculate_iou_cpu));
}

}