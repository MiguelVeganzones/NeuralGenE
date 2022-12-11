#ifndef ACTIVATION_FUNCTIONS
#define ACTIVATION_FUNCTIONS

#include <functional>

//--------------------------------------------------------------------------------------//

// namespace activation_functions
//{
//   template<typename T>
//   [[nodiscard]] inline static
//     T ReLU_impl(T x)
//   {
//     return std::max(x, T(0));
//   }
//
//   template<typename T>
//   [[nodiscard]] inline static
//     T Identity_impl(T x) {
//     return x;
//   }
//
//   template<typename T>
//     requires std::is_floating_point<T>::value
//   [[nodiscard]] inline static
//     T Sigmoid_impl(T x)
//   {
//     return static_cast<T>(T(1) / (T(1) + std::exp(-x)));
//   }
//
//   template<typename T>
//   [[nodiscard]] inline static
//     T Tanh_impl(T x)
//   {
//     return static_cast<T>(std::tanh(x));
//   }
//
//   enum Identifiers : int {
//     ReLU = 0,
//     Sigmoid,
//     Tanh,
//     Identity
//   };
//
//   template<typename T, activation_functions::Identifiers Function_Identifier>
//   [[nodiscard]]
//   constexpr
//     std::function<T(T)> choose_func()
//   {
//     if constexpr (Function_Identifier == activation_functions::ReLU)
//       return activation_functions::ReLU_impl<T>;
//     else if constexpr (Function_Identifier == activation_functions::Sigmoid)
//       return activation_functions::Sigmoid_impl<T>;
//     else if constexpr (Function_Identifier == activation_functions::Tanh)
//       return activation_functions::Tanh_impl<T>;
//     else if constexpr (Function_Identifier == activation_functions::Identity)
//       return activation_functions::Identity_impl<T>;
//   }
// }

namespace matrix_activation_functions
{
template <typename Mat>
using f_ptr = Mat (*)(Mat const&);

// ReSharper disable once CppInconsistentNaming
template <typename Mat>
[[nodiscard]] inline static Mat ReLU_impl(const Mat& mat)
{
    using T = typename Mat::value_type;
    return mat.apply([](T x) { return std::max(x, T(0)); });
}

template <typename Mat>
[[nodiscard]] inline static Mat Identity_impl(const Mat& mat)
{
    return mat;
}

template <typename Mat>
[[nodiscard]] inline static Mat Sigmoid_impl(const Mat& mat)
{
    using T = typename Mat::value_type;
    return mat.apply([](T x) { return static_cast<T>(T(1) / (T(1) + std::exp(-x))); });
}

template <typename Mat>
[[nodiscard]] inline static Mat Tanh_impl(const Mat& mat)
{
    using T = typename Mat::value_type;
    return mat.apply([](T x) { return static_cast<T>(std::tanh(x)); });
}

enum Identifiers
{
    ReLU = 0,
    Sigmoid,
    Tanh,
    Identity
};

template <typename Mat, matrix_activation_functions::Identifiers Function_Identifier>
// ReSharper disable once CppNotAllPathsReturnValue
[[nodiscard]] constexpr f_ptr<Mat> choose_func()
{
    if constexpr (Function_Identifier == matrix_activation_functions::ReLU)
        return matrix_activation_functions::ReLU_impl<Mat>;
    else if constexpr (Function_Identifier == matrix_activation_functions::Sigmoid)
        return matrix_activation_functions::Sigmoid_impl<Mat>;
    else if constexpr (Function_Identifier == matrix_activation_functions::Tanh)
        return matrix_activation_functions::Tanh_impl<Mat>;
    else if constexpr (Function_Identifier == matrix_activation_functions::Identity)
        return matrix_activation_functions::Identity_impl<Mat>;
    exit(EXIT_FAILURE);
}
} // namespace matrix_activation_functions

#endif // ACTIVATION_FUNCTIONS
