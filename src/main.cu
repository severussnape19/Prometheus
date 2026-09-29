#include "core/types.hpp"
#include "core/framebuffer.cuh"
#include "rendering/pixel.cuh"

auto main(i32 argc, char* argv[]) -> i32 {
    constexpr u32 HEIGHT = 600;
    constexpr u32 WIDTH  = 800;
    constexpr u32 N      = HEIGHT * WIDTH;

    Framebuffer fb(HEIGHT, WIDTH);

    dim3 block(16, 16);
    dim3 grid(
        (WIDTH + block.x - 1) / block.x,
        (HEIGHT + block.y - 1) / block.y
    );

    paint_image<<<grid, block>>>(fb);

    fb.generatePNG("image.png");

    return 0;
}
