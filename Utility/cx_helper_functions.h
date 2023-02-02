#pragma once

#ifndef CX_HELPER_FUNCTIONS
#define CX_HELPER_FUNCTIONS

#include <type_traits>

namespace cx_helper_func
{
template <class T>
    requires std::is_arithmetic_v<T>
[[nodiscard]] inline constexpr T cx_abs(const T& x) noexcept
{
    return x < 0 ? -x : x;
}

template <class T>
    requires std::is_arithmetic_v<T>
[[nodiscard]] inline constexpr T cx_max(T a, T b) noexcept
{
    return a < b ? b : a;
}

/**
 * \note Beware of overflow
 */
template <typename T1, typename T2>
    requires std::is_arithmetic_v<T1> && std::is_integral_v<T1> && std::is_arithmetic_v<T2> && std::is_integral_v<T2>
[[nodiscard]] inline constexpr T1 cx_pow(T1 base, T2 exp) noexcept
{
    auto ret = base;
    while (--exp)
        ret *= base;
    return ret;
}

[[nodiscard]] inline constexpr size_t cx_bits_required(size_t states) noexcept
{
    size_t n = 1;
    while (cx_pow<size_t, size_t>(2, n) < states)
        ++n;
    return n;
}

[[nodiscard]] inline constexpr size_t cx_pow2_bits_required(size_t states) noexcept
{
    size_t n = 1;
    while (cx_pow<size_t, size_t>(2, n) < states)
        n *= 2;
    return n;
}
} // namespace cx_helper_func

namespace helper_functions
{
template <typename T>
void pointer_swap(T* r, T* s)
{
    auto temp = *r;
    *r        = *s;
    *s        = temp;
}

} // namespace helper_functions
#endif // !CX_HELPER_FUNCTIONS
