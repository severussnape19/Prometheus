#pragma once
#include "../core/framebuffer.cuh"
#include "camera.cuh"
#include "../geometry/sphere.cuh"

__device__ __host__ auto ray_color(Ray const& r, Sphere const& sphere) -> Color {
    if (sphere.hit(r)) {
        return Color(1.f, 0.f, 0.f);
    }

    Vec3f unit_direction = r.direction().normalized();
    auto t = 0.5f * (unit_direction.y + 1.0f);
    return lerp(Color(1.0f, 1.0f, 1.0f), Color(0.3f, 0.7f, 1.0f), t);
}

__global__ auto paint_image(Camera& camera, Framebuffer& fb, Sphere const& sphere) -> void {
    u32 i = blockDim.x * blockIdx.x + threadIdx.x;
    u32 j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= fb.width() || j >= fb.height())
        return;

    u32 pixel_index = j * fb.width() + i;

    auto pixel_center = camera.get00pxLoc() +
        (static_cast<f32>(i) * camera.pixel_du()) + (static_cast<f32>(j) * camera.pixel_dv());

    auto ray_direction = pixel_center - camera.getCenter();
    Ray ray(camera.getCenter(), ray_direction);

    auto color = ray_color(ray, sphere);

    auto r = color.x;
    auto g = color.y;
    auto b = color.z;

    fb[pixel_index].x = r;
    fb[pixel_index].y = g;
    fb[pixel_index].z = b;
}
