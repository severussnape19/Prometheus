#pragma once

#include "math.hpp"
#include "device_buffer.cuh"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include "helper.cuh"

struct Pixel {
    u8 r, g, b;
};

struct [[nodiscard]] Framebuffer {
public:
    u32 height_{}, width_{};

    Framebuffer(u32 image_height, u32 image_width)
        : height_(image_height)
        , width_(image_width)
    {
        buffer_ = DeviceBuffer(image_width * image_height);
    }

    auto generatePNG(char const* filename) -> bool {
        std::vector<Color> buf(height_ * width_);
        cuda_check(cudaMemcpy(buf.data(), buffer_.getData(), height_ * width_ * sizeof(Color), cudaMemcpyDeviceToHost));
        std::vector<u8> pixels(static_cast<size_t>(width_ * height_ * 3));

        auto toByte = [](f32 x) -> u8 {
            x = std::clamp(x, 0.0f, 1.0f);
            return static_cast<u8>(x * 255.0f + 0.5f);
        };

        auto linearToSRGB = [](f32 x) -> f32 {
            if (x <= 0.0031308f) {
                return 12.92f * x;
            }
            return 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
        };

        for (usize i = 0; i < static_cast<usize>(width_ * height_); ++i) {
            pixels[i * 3 + 0] = toByte(linearToSRGB(buf[i].x));
            pixels[i * 3 + 1] = toByte(linearToSRGB(buf[i].y));
            pixels[i * 3 + 2] = toByte(linearToSRGB(buf[i].z));
        }

        return stbi_write_png(
            filename,
            width_,
            height_,
            3,
            pixels.data(),
            width_ * 3) != 0;
    }

    __host__ __device__ auto operator[](u32 index) -> Color& {
        return buffer_.getData()[index];
    }
private:
    // AoS for now
    DeviceBuffer buffer_;
};
