#include "core/types.hpp"
#include "rendering/pixel.cuh"
#include "rendering/camera.cuh"
#include "geometry/sphere.cuh"

constexpr u32 WIDTH = 800;

auto main(i32 argc, char* argv[]) -> i32 {
    Camera camera(static_cast<f32>(16.f / 9.f), WIDTH, 1.f);

    Framebuffer fb(camera.getHeight(), camera.getWidth());

    dim3 block(16, 16);
    dim3 grid(
        (fb.width() + block.x - 1) / block.x,
        (fb.height() + block.y - 1) / block.y
    );

    Sphere sphere(Point(0.f, 0.f, -1.f), -0.5f);

    paint_image<<<grid, block>>>(camera, fb, sphere);

    fb.generatePNG("image.png");

    return 0;
}
