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
} // namespace cx_helper_func

#endif // !CX_HELPER_FUNCTIONS
