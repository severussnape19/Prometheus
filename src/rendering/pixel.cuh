#pragma once
#include "../core/framebuffer.cuh"

__global__ auto paint_image(Framebuffer& fb) -> void {
    u32 i = gridDim.x * blockIdx.x + threadIdx.x;
    u32 j = gridDim.y * blockIdx.y + threadIdx.y;
    u32 pixel_index = j * fb.width_ + i;

    if (i < fb.width_ && j < fb.height_) {
        auto r = static_cast<f32>(i) / (fb.width_ - 1);
        auto g = static_cast<f32>(j) / (fb.height_ - 1);
        auto b = (f32)0.0;

        fb[pixel_index].x = static_cast<i32>(255.999 * r);
        fb[pixel_index].y = static_cast<i32>(255.999 * g);
        fb[pixel_index].z = static_cast<i32>(255.999 * b);
    }
}
