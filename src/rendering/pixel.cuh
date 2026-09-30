#pragma once
#include "../core/framebuffer.cuh"

__global__ auto paint_image(Framebuffer& fb) -> void {
    u32 i = blockDim.x * blockIdx.x + threadIdx.x;
    u32 j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= fb.width() && j >= fb.height())
        return;

    u32 pixel_index = j * fb.width() + i;

    auto r = static_cast<f32>(i) / (fb.width() - 1);
    auto g = static_cast<f32>(j) / (fb.height() - 1);
    auto b = (f32)0.0;

    fb[pixel_index].x = r;
    fb[pixel_index].y = g;
    fb[pixel_index].z = b;
}
