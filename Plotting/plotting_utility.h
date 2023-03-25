
#ifndef PLOTTING_UTILITY_EVOLUTION_ENV
#define PLOTTING_UTILITY_EVOLUTION_ENV

#include <cassert>

#if defined(__clang__)
#pragma clang diagnostic push
#endif

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#endif

#include <matplot/matplot.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif


#include <array>
#include <cmath>
#include <concepts>
#include <iostream>
#include <numeric>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

namespace plotting_utility
{

class plot_formatting
{
    inline static const std::vector<std::string> s_format_configs = { ":", "--", "-", "-o", "-*" };

    inline static unsigned int s_idx = 0;

public:
    inline static std::string_view get()
    {
        const auto i = s_idx++;
        s_idx %= s_format_configs.size();
        return s_format_configs[i];
    }

    inline static void reset()
    {
        s_idx = 0;
    }
};

// iterator concept
template <typename Vec>
concept iterable_type = requires(Vec& v)
{
    // std::ranges::range<Vec>;
    v.begin();
    v.end();
    // Vec::value_type;
    v.size();
};

template <iterable_type Vec>
void plot(const Vec& y)
{
    using T = typename Vec::value_type;

    const auto N = y.size();

    std::vector<T> vx(N);
    std::vector<T> vy(N);

    std::iota(std::begin(vx), std::end(vx), 0.);
    std::copy(std::begin(y), std::end(y), std::begin(vy));

    matplot::plot(vx, vy, "-o");

    matplot::show();
}

template <iterable_type Vec>
void plot(const Vec& x, const std::vector<Vec>& y_)
{
    using T = typename Vec::value_type;

    const auto N = x.size();
    for (const auto& e : y_)
        assert(e.size() == N);

    matplot::hold(matplot::off);

    std::vector<T> vx(N);
    std::iota(std::begin(vx), std::end(vx), 0);

    for (const auto& y : y_)
    {
        std::vector<T> vy(N);
        std::copy(y.begin(), y.end(), std::begin(vy));
        matplot::plot(vx, vy, plot_formatting::get());
        matplot::hold(matplot::on);
    }

    matplot::show();
    matplot::hold(matplot::off);
    plot_formatting::reset();
}

} // namespace plotting_utility

#endif // PLOTTING_UTILITY_EVOLUTION_ENV
