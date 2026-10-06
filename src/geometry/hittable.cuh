#pragma once
#include "../core/math.cuh"
#include "ray.cuh"
#include <cmath>
#include "../core/utilities.cuh"

class HitRecord {
public:
    Point p;
    Vec3f normal;
    f32 t;
    bool front_face;

    HD auto set_face_normal(Ray const& ray, Vec3f const& outward_normal) -> void {
        // outward_normal is assumed to be normalized
        front_face = ray.direction().dot(outward_normal) < 0; // -ve if both are opposite to eachother else +ve
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Hittable {
public:
    virtual ~Hittable() = default;
    HD virtual auto hit(Ray const& ray, Interval ray_t, HitRecord& rec) const -> bool = 0;
};

class Sphere : public Hittable {
public:
    HD explicit Sphere(Point const& center, f32 radius)
        : center_(center)
        , radius_(radius) {}

    HD auto hit(Ray const& ray, Interval ray_t, HitRecord& rec) const -> bool override {
        Vec3f ray_dir = ray.direction();
        Vec3f L = ray.origin() - center_;

        f32 a = ray_dir.dot(ray_dir);
        f32 b = 2.0f * ray_dir.dot(L);
        f32 c = L.dot(L) - (radius_ * radius_);

        f32 delta = (b * b) - (4 * a * c);
        if (delta < 0) {
            return false;
        }

        f32 sqrt_delta  = std::sqrt(delta);
        f32 denominator = 2.0f * a;
        auto root = (-b - sqrt_delta) / denominator;
        if (!ray_t.surrounds(root)) { // !(root < ray_t.max || root > ray_t.min)
            root = (-b + sqrt_delta) / denominator;
            if (!ray_t.surrounds(root)) {
                return false;
            }
        }

        rec.t = root;
        rec.p = ray.at(rec.t); // hit point
        Vec3f outward_normal = (rec.p - center_) / radius_; // normalized outward normal
        rec.set_face_normal(ray, outward_normal);

        return true;
    }

    [[nodiscard]] HD auto origin() const -> Point { return center_; }
    [[nodiscard]] HD auto radius() const -> f32 { return radius_; }
private:
    Point center_;
    f32 radius_;
};

__global__ auto create_sphere(Sphere* sphere, Point center, f32 radius) -> void {
    new (sphere) Sphere(center, radius);
}

class HittableList {
public:
    Hittable** objects{};
    usize object_count{};

    HittableList() = default;

    HD auto hit(Ray const& ray, Interval ray_t, HitRecord& rec) const -> bool {
        HitRecord temp_rec{};
        bool hit_anything{};
        f32 closest_so_far = ray_t.getMax();

        for (usize i{}; i < object_count; i++) {
            if (objects[i]->hit(ray, Interval(ray_t.getMin(), closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }
};
