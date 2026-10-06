#pragma once
#include "types.hpp"
#include "math.cuh"

constexpr auto infinity = std::numeric_limits<f32>::infinity();
constexpr auto pi       = 3.1415926535897932385f;

HD constexpr inline auto degree_to_radians(f32 degrees) -> f32 {
    return degrees * pi / 180.f;
}

class Interval {
public:
    HD Interval()
        : min(infinity)
        , max(-infinity) {}
    HD Interval(f32 min, f32 max)
        : min(min)
        , max(max) {}

    [[nodiscard]] HD auto size()           const -> f32  { return max - min; }
    [[nodiscard]] HD auto contains(f32 x)  const -> bool { return x >= min && x <= max; }
    [[nodiscard]] HD auto surrounds(f32 x) const -> bool { return x > min && x < max; }

    [[nodiscard]] HD auto getMin() const -> f32 { return min; }
    [[nodiscard]] HD auto getMax() const -> f32 { return max; }

    static const Interval empty, universe;
private:
    f32 min{}, max{};
};

const Interval Interval::empty = Interval(+infinity, -infinity);
const Interval Interval::universe = Interval(-infinity, +infinity);
