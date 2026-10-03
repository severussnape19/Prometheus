#pragma once
#include "../core/math.cuh"
#include "../core/types.hpp"

class Ray {
public:
    Ray() = default;
    __device__ __host__
    explicit Ray(Point const& origin, Vec3f const& direction)
        : origin_(origin)
        , direction_(direction) {}

    [[nodiscard]] __host__ __device__ auto origin()    const -> Point const& { return origin_; }
    [[nodiscard]] __host__ __device__ auto direction() const -> Vec3f const& { return direction_; }
    [[nodiscard]] __host__ __device__ auto at(f32 t)   const -> Point { return origin_ + (t * direction_); }
private:
    Point origin_;
    Vec3f direction_;
};
