#pragma once
#include "../core/math.cuh"
#include "ray.cuh"

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

class Sphere {
public:
    HD explicit Sphere(Point const& center, f32 radius)
        : center_(center)
        , radius_(radius) {}

    HD auto hit(Ray const& ray) const -> bool {
        Vec3f ray_dir = ray.direction();
        Vec3f L = ray.origin() - center_;
        Vec3f oc = center_ - ray_dir;

        f32 a = ray_dir.dot(ray_dir);
        f32 b = 2.0f * ray_dir.dot(L);
        f32 c = L.dot(L) - (radius_ * radius_);

        f32 delta = (b * b) - (4 * a * c);

        return delta >= 0 ? true : false;
    }
private:
    Point center_;
    f32 radius_;
};
