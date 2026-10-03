#pragma once
#include "types.hpp"

template <typename T>
struct DeviceShared_ptr {
    T* ptr{};
    usize* ref_count{};
};
