#pragma once
#include "../core/math.cuh"
#include "../core/types.hpp"

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

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
