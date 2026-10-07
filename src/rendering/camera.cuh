#pragma once
#include "../core/math.cuh"
#include "../core/types.hpp"
#include "../geometry/hittable.cuh"
#include "../core/framebuffer.cuh"

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

HD auto ray_color(Ray const &r, HittableList const &hit_list) -> Color {
    HitRecord hit_rec{};
    Interval interval(0.001, infinity);
    if (hit_list.hit(r, interval, hit_rec)) {
        return 0.5f * Color(hit_rec.normal + Color(1.f, 1.f, 1.f));
    }

    Vec3f unit_direction = r.direction().normalized();
    auto a = 0.5f * (unit_direction.y + 1.0f);
    return lerp(Color(1.0f, 1.0f, 1.0f), Color(0.3f, 0.7f, 1.0f), a);
}

struct Camera {
public:
    Camera() = default;
    explicit Camera(f32 aspect_ratio, u32 width, f32 focal_length)
        : width_(width)
        , height_(static_cast<u32>(width / aspect_ratio))
        , aspect_ratio(aspect_ratio)
        , focal_length(focal_length)
        , viewport_height_(2.0)
        , viewport_width_(viewport_height_ * (f32(width_) / height_))
    {
        viewport_u = Vec3f(viewport_width_, 0, 0);
        viewport_v = Vec3f(0, -viewport_height_, 0);

        px_du = viewport_u / width_;
        px_dv = viewport_v / height_;

        upper_left = center - Vec3f(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = upper_left + (0.5f * (px_du + px_dv));
    }

    [[nodiscard]] HD auto getViewportWidth() const -> f32 const {
      return viewport_width_;
    }
    [[nodiscard]] HD auto getViewportHeight() const -> f32 const {
      return viewport_height_;
    }

    [[nodiscard]] HD auto getWidth() const -> u32 const { return width_; }
    [[nodiscard]] HD auto getHeight() const -> u32 const { return height_; }
    [[nodiscard]] HD auto get00pxLoc() const -> Point const { return pixel00_loc; }
    [[nodiscard]] HD auto pixel_du()   const -> Point const { return px_du; }
    [[nodiscard]] HD auto pixel_dv()   const -> Point const { return px_dv; }
    [[nodiscard]] HD auto getCenter()  const -> Point const { return center; }

    auto render_cpu(Framebuffer_host &fb, HittableList const& world) -> void {
        for (usize j{}; j < fb.height(); ++j) {
            for (usize i{}; i < fb.width(); ++i) {
                auto pixel_center = pixel00_loc +
                              static_cast<f32>(i) * px_du +
                              static_cast<f32>(j) * px_dv;

                auto ray_direction = pixel_center - center;
                Ray ray(center, ray_direction);

                Color color = ray_color(ray, world);

                auto r = color.x;
                auto g = color.y;
                auto b = color.z;

                usize pixel_index = j * fb.width() + i;

                fb[pixel_index].x = r;
                fb[pixel_index].y = g;
                fb[pixel_index].z = b;
            }
        }
    }
private:
    Point center = Point(0.f, 0.f, 0.f);
    Point pixel00_loc;
    u32 width_{}, height_{};
    f32 viewport_height_{}, viewport_width_{};
    f32 aspect_ratio{}, focal_length{};
    Vec3f viewport_u, viewport_v;
    Vec3f px_du, px_dv;
    Vec3f upper_left;
};

__global__ auto paint_image(Camera &camera, Framebuffer_device &fb, HittableList const &hit_list) -> void {
    u32 i = blockDim.x * blockIdx.x + threadIdx.x;
    u32 j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= fb.width() || j >= fb.height())
      return;

    u32 pixel_index = j * fb.width() + i;

    auto pixel_center = camera.get00pxLoc() +
                        (static_cast<f32>(i) * camera.pixel_du()) +
                        (static_cast<f32>(j) * camera.pixel_dv());

    auto ray_direction = pixel_center - camera.getCenter();
    Ray ray(camera.getCenter(), ray_direction);

    auto color = ray_color(ray, hit_list);

    auto r = color.x;
    auto g = color.y;
    auto b = color.z;

    fb[pixel_index].x = r;
    fb[pixel_index].y = g;
    fb[pixel_index].z = b;
}
