#ifndef ACTIVATION_FUNCTIONS
#define ACTIVATION_FUNCTIONS

#include <functional>
#include <stdexcept>
#include <cmath>

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
    return mat.apply_fn([](T x) { return std::max(x, T{}); }); //{ return x > T{} ? x : T{}; });
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
    return mat.apply_fn([](T x) { return static_cast<T>(T(1) / (T(1) + std::exp(-x))); });
}

template <typename Mat>
[[nodiscard]] inline static Mat Tanh_impl(const Mat& mat)
{
    using T = typename Mat::value_type;
    return mat.apply_fn([](T x) { return static_cast<T>(std::tanh(x)); });
}

/**
 * \brief Gausian error linear unit activation function
 * \tparam Mat 
 * \param mat 
 * \return 
 */
template <typename Mat>
[[nodiscard]] inline static Mat GELU_impl(const Mat& mat)
{
    using T = typename Mat::value_type;
    return mat.apply_fn([](T x) { return static_cast<T>(x / 2 * (T(1) + std::erf(x / std::sqrt(T(2))))); });
}

/**
 * \brief Sigmoid linear unit activation function
 * \tparam  
 * \param  
 * \return 
 */
template <typename Mat>
[[nodiscard]] inline static Mat SiLU_impl(const Mat& mat)
{
    using T = typename Mat::value_type;
    return mat.apply_fn([](T x) { return static_cast<T>(x / (T(1) + std::exp(-x))); });
}

enum Identifiers
{
    ReLU = 0,
    Sigmoid,
    Tanh,
    Identity,
    GELU,
    SiLU
};

template <typename Mat, matrix_activation_functions::Identifiers Function_Identifier>
// ReSharper disable once CppNotAllPathsReturnValue
[[nodiscard]] constexpr f_ptr<Mat> choose_func()
{
    if constexpr (Function_Identifier == matrix_activation_functions::Identifiers::ReLU)
        return matrix_activation_functions::ReLU_impl<Mat>;
    else if constexpr (Function_Identifier == matrix_activation_functions::Identifiers::Sigmoid)
        return matrix_activation_functions::Sigmoid_impl<Mat>;
    else if constexpr (Function_Identifier == matrix_activation_functions::Identifiers::Tanh)
        return matrix_activation_functions::Tanh_impl<Mat>;
    else if constexpr (Function_Identifier == matrix_activation_functions::Identifiers::Identity)
        return matrix_activation_functions::Identity_impl<Mat>;
    else if constexpr (Function_Identifier == matrix_activation_functions::Identifiers::GELU)
        return matrix_activation_functions::GELU_impl<Mat>;
    else if constexpr (Function_Identifier == matrix_activation_functions::Identifiers::SiLU)
        return matrix_activation_functions::SiLU_impl<Mat>;
    throw std::invalid_argument("Unexpected activation function identifier");
}
} // namespace matrix_activation_functions

#endif // ACTIVATION_FUNCTIONS
