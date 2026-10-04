#pragma once
#include "../core/framebuffer.cuh"
#include "../geometry/hittable.cuh"
#include "camera.cuh"
#include "../core/math.cuh"

__device__ __host__ auto ray_color(Ray const &r, HittableList const &hit_list)
    -> Color {
  HitRecord hit_rec{};
  if (hit_list.hit(r, 0.f, INFINITY, hit_rec)) {
    return 0.5f * Color(hit_rec.normal + Color(1.f, 1.f, 1.f));
  }

  Vec3f unit_direction = r.direction().normalized();
  auto a = 0.5f * (unit_direction.y + 1.0f);
  return lerp(Color(1.0f, 1.0f, 1.0f), Color(0.3f, 0.7f, 1.0f), a);
}

__global__ auto paint_image(Camera &camera, Framebuffer &fb,
                            HittableList const &hit_list) -> void {
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

auto paint_host_image(Camera const &camera, Framebuffer_host &fb,
                      HittableList const &hit_list) -> void {
  for (usize j{}; j < fb.height(); ++j) {
    for (usize i{}; i < fb.width(); ++i) {
      auto pixel_center = camera.get00pxLoc() +
                          static_cast<f32>(i) * camera.pixel_du() +
                          static_cast<f32>(j) * camera.pixel_dv();

      auto ray_direction = pixel_center - camera.getCenter();
      Ray ray(camera.getCenter(), ray_direction);

      Color color = ray_color(ray, hit_list);

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
