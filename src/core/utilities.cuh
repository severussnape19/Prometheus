#pragma once
#include "types.hpp"

constexpr auto infinity = std::numeric_limits<f32>::infinity;
constexpr auto pi       = 3.1415926535897932385f;

#ifdef __CUDACC__
__device__ __host__
#endif
constexpr inline auto degree_to_radians(f32 degrees) -> f32 {
    return degrees * pi / 180.f;
}
