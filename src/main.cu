#include "core/types.hpp"
#include "geometry/hittable.cuh"
#include "rendering/camera.cuh"
#include "rendering/render.cuh"

constexpr u32 WIDTH = 1280;

auto main(i32 argc, char* argv[]) -> i32 {
    Camera camera(static_cast<f32>(16.f / 9.f), WIDTH, 1.f);

    Sphere *d_surface{};
    Sphere *d_sphere{};
    cuda_check(cudaMalloc(&d_surface, sizeof(Sphere)));
    cuda_check(cudaMalloc(&d_sphere, sizeof(Sphere)));

    create_sphere<<<1, 1>>>(d_sphere, Point(0.f, 0.f, -1.f), 0.5f);
    create_sphere<<<1, 1>>>(d_surface, Point(0.f, -100.5f, -1.f), 100.f);

    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());

    Hittable **d_objects{};
    cuda_check(cudaMalloc(&d_objects, 2 * sizeof(Hittable *)));

    std::array<Hittable *, 2> objects = {d_sphere, d_surface};
    cuda_check(cudaMemcpy(d_objects, objects.data(), 2 * sizeof(Hittable *),
                          cudaMemcpyHostToDevice));

    HittableList world{};
    world.objects = d_objects;
    world.object_count = 2;

    Framebuffer fb(camera.getHeight(), camera.getWidth());

    dim3 block(16, 16);
    dim3 grid(
        (fb.width() + block.x - 1) / block.x,
        (fb.height() + block.y - 1) / block.y
    );

    paint_image<<<grid, block>>>(camera, fb, world);

    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());

    fb.generatePNG("image.png");

    HittableList world_host{};

    Sphere sphere(Point(0.f, 0.f, -1.f), 0.5f);
    Sphere surface(Point(0.f, -100.5f, -1.f), 100.f);

    std::array<Hittable *, 2> host_objs = {&sphere, &surface};
    world_host.objects = host_objs.data();
    world_host.object_count = 2;

    Framebuffer fb_host(camera.getHeight(), camera.getWidth());

    paint_host_image(camera, fb_host, world_host);

    fb_host.generatePNG("image_cpu.png");

    cuda_check(cudaFree(d_objects));
    cuda_check(cudaFree(d_sphere));
    cuda_check(cudaFree(d_surface));

    return 0;
}
