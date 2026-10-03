#pragma once
#include "math.cuh"
#include "types.hpp"
#include <utility>

struct DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(usize n) : count(n) {
        cudaMalloc(&data, count * sizeof(Color));
    }

    explicit DeviceBuffer(void const* src, usize count, enum cudaMemcpyKind kind)
        : count(count)
    {
        cudaMemcpy(data, src, count, kind);
    }

    DeviceBuffer(DeviceBuffer&& o) noexcept
        : data(std::exchange(o.data, nullptr))
        , count(std::exchange(o.count, 0u)) {}

    DeviceBuffer(DeviceBuffer const&) = delete;
    auto operator=(DeviceBuffer const&) = delete;

    auto operator=(DeviceBuffer&& o) -> DeviceBuffer& {
        if (this == &o) return *this;

        release();

        data = std::exchange(o.data, nullptr);
        count = std::exchange(o.count, 0u);

        return *this;
    }

    ~DeviceBuffer() {
        if (data) {
            release();
        }
    }

    auto copyFrom(void const* src, usize n, enum cudaMemcpyKind const kind) -> void {
        cudaMemcpy(data, src, n, kind);
    }

    [[nodiscard]] __host__ __device__ auto getData() -> Color* { return data; }
    [[nodiscard]] auto getCount() const -> usize { return count; }
private:
    auto release() noexcept -> void {
        if (data) cudaFree(data);
        data = nullptr;
        count = 0;
    }

    Color* data{};
    usize count{};
};
