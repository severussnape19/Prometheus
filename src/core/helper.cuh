#pragma once

#include <source_location>
#include <cstdio>

auto log(
    char const* msg,
    std::source_location loc = std::source_location::current()
) -> void {
    fprintf(stderr, "%s:%d %s", loc.file_name(), loc.line(), msg);
}

inline auto cuda_check(cudaError_t err, std::source_location loc = std::source_location::current()) -> void {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s:%d %s", loc.file_name(), loc.line(), cudaGetErrorString(err));
    }
}
