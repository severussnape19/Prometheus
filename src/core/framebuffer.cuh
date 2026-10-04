#pragma once

#include "math.cuh"
#include "device_buffer.cuh"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include "helper.cuh"
#include <fstream>

auto toByte(f32 x) -> u8 {
    x = std::clamp(x, 0.0f, 1.0f);
    return static_cast<u8>(x * 255.0f + 0.5f);
};

auto linearToSRGB(f32 x) -> f32 {
    if (x <= 0.0031308f) {
        return 12.92f * x;
    }
    return 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
};

struct [[nodiscard]] Framebuffer {
public:
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
    [[nodiscard]] __host__ __device__ auto height() const -> u32 const { return height_; }
    [[nodiscard]] __host__ __device__ auto width()  const -> u32 const { return width_; }
private:
    // AoS for now
    DeviceBuffer buffer_;
    u32 height_{}, width_{};
};

struct [[nodiscard]] Framebuffer_host {
public:
    explicit Framebuffer_host(u32 height, u32 width)
        : height_(height), width_(width)
    {
        buffer_.resize(height * width);
    }

    auto generatePNG(char const* filename) -> bool {
        std::vector<u8> pixels(buffer_.size() * 3);
        for (usize i{}; i < buffer_.size(); ++i) {
            pixels[i * 3 + 0] = toByte(linearToSRGB(buffer_[i].x));
            pixels[i * 3 + 1] = toByte(linearToSRGB(buffer_[i].y));
            pixels[i * 3 + 2] = toByte(linearToSRGB(buffer_[i].z));
        }
        return stbi_write_png(
            filename,
            width_,
            height_,
            3,
            pixels.data(),
            width_ * 3) != 0;
    }

    auto generatePPM(char const* filename) -> void {
        std::ofstream outfile(filename);

        if (!outfile.is_open()) {
            throw std::runtime_error("Could not open file!");
        }

        outfile << "P3\n" << width_ << ' ' << height_ << "\n255\n";
        for (usize i{}; i < buffer_.size(); ++i) {
            usize pixel_index = i;

            u32 r = static_cast<u32>(buffer_[pixel_index].x * 255.999f);
            u32 g = static_cast<u32>(buffer_[pixel_index].y * 255.999f);
            u32 b = static_cast<u32>(buffer_[pixel_index].z * 255.999f);

            outfile << r << ' ' << g << ' ' << b << '\n';
        }

        outfile.close();
    }

    [[nodiscard]] auto operator[](usize index) -> Color& {
        return buffer_[index];
    }

    [[nodiscard]] auto height() const -> u32 const { return height_; }
    [[nodiscard]] auto width()  const -> u32 const { return width_; }
private:
  std::vector<Color> buffer_;
  u32 height_{}, width_{};
};
