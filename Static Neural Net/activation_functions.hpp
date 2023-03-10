#ifndef ACTIVATION_FUNCTIONS
#define ACTIVATION_FUNCTIONS

#include <functional>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

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
// ReSharper disable once CppInconsistentNaming
template <typename Mat>
struct ReLU
{
    using T = typename Mat::value_type;

    struct params
    {
        [[nodiscard]] static constexpr size_t parameter_count()
        {
            return 0;
        }

        [[nodiscard]] constexpr auto repr() const -> const char*
        {
            return "";
        }

        template <typename Fn>
        void mutate(Fn&& fn)
        {
        }

        template <typename Fn, typename... Args>
            requires std::is_invocable_r_v<T, Fn, Args...>
        constexpr void fill(Fn&& fn, Args... args)
        {
        }
    } m_params;

    using parameters_type = params;

    inline void operator()(Mat& mat) const
    {
        std::invoke(ReLU_impl, mat);
    }

    inline static void ReLU_impl(Mat& mat)
    {
        mat.transform([](T x) { return x > T{} ? x : T{}; }); // { return std::max(T{}, x); }
    }

    static constexpr auto name = "ReLU";
};

// ReSharper disable once CppInconsistentNaming
template <typename Mat>
struct Sigmoid
{
    using T = typename Mat::value_type;

    struct params
    {
        [[nodiscard]] static constexpr size_t parameter_count()
        {
            return 0;
        }

        [[nodiscard]] constexpr auto repr() const -> const char*
        {
            return "";
        }

        template <typename Fn>
        void mutate(Fn&& fn)
        {
        }

        template <typename Fn, typename... Args>
            requires std::is_invocable_r_v<T, Fn, Args...>
        constexpr void fill(Fn&& fn, Args... args)
        {
        }
    } m_params;

    using parameters_type = params;

    inline void operator()(Mat& mat) const
    {
        std::invoke(Sigmoid_impl, mat);
    }

     inline static void Sigmoid_impl(Mat& mat)
     {
         mat.transform([](T x) { return static_cast<T>(T(1) / (T(1) + std::exp(-x))); });
     }

    static constexpr auto name = "Sigmoid";
};

template <typename Mat>
struct Identity
{
    using T = typename Mat::value_type;

    struct params
    {
         [[nodiscard]] static constexpr size_t parameter_count()
        {
            return 0;
        }

        [[nodiscard]] constexpr auto repr() const -> const char*
        {
            return "";
        }

        template <typename Fn>
        void mutate(Fn&& fn)
        {
        }

        template <typename Fn, typename... Args>
            requires std::is_invocable_r_v<T, Fn, Args...>
        constexpr void fill(Fn&& fn, Args... args)
        {
        }
    } m_params;

    using parameters_type = params;

    inline void operator()(Mat& mat) const {
        std::invoke(Identity_impl, mat);
    }

    inline static void Identity_impl(Mat& mat)
    {
    }

    static constexpr auto name = "Identity";
};

template <typename Mat>
struct Tanh
{
    using T = typename Mat::value_type;

    struct params
    {
        [[nodiscard]] static constexpr size_t parameter_count()
        {
            return 0;
        }

        [[nodiscard]] constexpr auto repr() const -> const char*
        {
            return "";
        }

        template <typename Fn>
        void mutate(Fn&& fn)
        {
        }

        template <typename Fn, typename... Args>
            requires std::is_invocable_r_v<T, Fn, Args...>
        constexpr void fill(Fn&& fn, Args... args)
        {
        }
    } m_params;

    using parameters_type = params;

    inline void operator()(Mat& mat) const
    {
        std::invoke(Tanh_impl, mat);
    }

    inline static void Tanh_impl(Mat& mat)
    {
        mat.transform([](T x) { return static_cast<T>(std::tanh(x)); });
    }

    static constexpr auto name = "Tanh";
};

/**
 * \brief Gausian error linear unit activation function
 */
template <typename Mat>
struct GELU
{
    using T = typename Mat::value_type;

    struct params
    {
        [[nodiscard]] static constexpr size_t parameter_count()
        {
            return 0;
        }

        [[nodiscard]] constexpr auto repr() const -> const char*
        {
            return "";
        }

        template <typename Fn>
        void mutate(Fn&& fn)
        {
        }

        template <typename Fn, typename... Args>
            requires std::is_invocable_r_v<T, Fn, Args...>
        constexpr void fill(Fn&& fn, Args... args)
        {
        }
    } m_params;

    using parameters_type = params;

    inline void operator()(Mat& mat) const
    {
        std::invoke(GELU_impl, mat);
    }

    inline static void GELU_impl(Mat& mat)
    {
        mat.transform([](T x) { return static_cast<T>(x / 2 * (T(1) + std::erf(x / std::sqrt(T(2))))); });
    }

    static constexpr auto name = "GELU";
};


/**
 * \brief Sigmoid linear unit activation function
 */
template <typename Mat>
struct SiLU
{
    using T = typename Mat::value_type;

    struct params
    {
        [[nodiscard]] static constexpr size_t parameter_count()
        {
            return 0;
        }

        [[nodiscard]] constexpr auto repr() const -> const char*
        {
            return "";
        }

        template <typename Fn>
        void mutate(Fn&& fn)
        {
        }

        template <typename Fn, typename... Args>
            requires std::is_invocable_r_v<T, Fn, Args...>
        constexpr void fill(Fn&& fn, Args... args)
        {
        }
    } m_params;

    using parameters_type = params;

    inline void operator()(Mat& mat) const
    {
        std::invoke(SiLU_impl, mat);
    }

    inline static void SiLU_impl(Mat& mat)
    {
        mat.transform([](T x) { return static_cast<T>(x / (T(1) + std::exp(-x))); });
    }

    static constexpr auto name = "SiLU";
};

/**
 * \brief Swish linear unit activation function
 */
template <typename Mat>
struct Swish
{
    using T = typename Mat::value_type;

    struct params
    {
        T beta;

        [[nodiscard]] static constexpr size_t parameter_count()
        {
            return 1;
        }

        [[nodiscard]] constexpr auto repr() const -> std::string
        {
            return std::string("Beta: ") + std::to_string(beta);
        }

        template <typename Fn>
        void mutate(Fn&& fn)
        {
            // todo enforce restrictions restrictions
            beta = fn(beta); 
        }

        template <typename Fn, typename... Args>
            requires std::is_invocable_r_v<T, Fn, Args...>
        constexpr void fill(Fn&& fn, Args... args)
        {
            // todo enforce restrictions
            beta = std::invoke(fn, args...);
        }
    } m_params;

    using parameters_type = params;

    inline void operator()(Mat& mat) const
    {
        Swish_impl(mat, m_params.beta);
    }

    inline static void Swish_impl(Mat& mat, T beta)
    {
        mat.transform([_beta=beta](T x) { return static_cast<T>(x / (T(1) + std::exp(-_beta*x))); });
    }

    static constexpr auto name = "Swish";
};

struct Identifiers
{
    enum Identifiers_
    {
         ReLU,
         Sigmoid,
         Tanh,
         Identity,
         GELU,
         SiLU,
         Swish
    };
};

template <typename Mat, matrix_activation_functions::Identifiers::Identifiers_ Function_Identifier>
// ReSharper disable once CppNotAllPathsReturnValue
[[nodiscard]] constexpr auto choose_func()
{
    if constexpr (Function_Identifier == matrix_activation_functions::Identifiers::ReLU)
        return matrix_activation_functions::ReLU<Mat>();
    else if constexpr (Function_Identifier == matrix_activation_functions::Identifiers::Sigmoid)
        return matrix_activation_functions::Sigmoid<Mat>();
    else if constexpr (Function_Identifier == matrix_activation_functions::Identifiers::Tanh)
        return matrix_activation_functions::Tanh<Mat>();
    else if constexpr (Function_Identifier == matrix_activation_functions::Identifiers::Identity)
        return matrix_activation_functions::Identity<Mat>();
    else if constexpr (Function_Identifier == matrix_activation_functions::Identifiers::GELU)
        return matrix_activation_functions::GELU<Mat>();
    else if constexpr (Function_Identifier == matrix_activation_functions::Identifiers::SiLU)
        return matrix_activation_functions::SiLU<Mat>();
    else if constexpr (Function_Identifier == matrix_activation_functions::Identifiers::Swish)
        return matrix_activation_functions::Swish<Mat>();
    std::unreachable();
}

template <typename Mat, matrix_activation_functions::Identifiers::Identifiers_ Function_Identifier>
class activation_function
{
public:
    inline static constexpr auto s_impl = choose_func<Mat, Function_Identifier>();

    using activation_function_type = decltype(s_impl);
    using parameters_type = typename activation_function_type::parameters_type;

    parameters_type params;

    void operator()(Mat& mat) const {
        std::invoke(s_impl, mat);
    }

    template<typename Fn>
    void mutate_params(Fn&& fn)
    {
        params.mutate(fn);
    }

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<typename activation_function_type::T, Fn, Args...>
    constexpr void fill(Fn&& fn, Args... args)
    {
        params.fill(fn, args...);
    }

    static constexpr size_t parameter_count()
    {
        return parameters_type::parameter_count();
    }

    [[nodiscard]] auto repr() const
    {
        std::stringstream ss{};
        ss << activation_function_type::name << '\t' << params.repr();
        return ss.str();
    }
};


// -----------------------------------------------
// Activation function concept
// -----------------------------------------------

template <typename Mat, matrix_activation_functions::Identifiers::Identifiers_ Function_Identifier>
void activation_function_dummy(activation_function<Mat, Function_Identifier>)
{
}

template <typename T>
concept activation_function_type = requires { activation_function_dummy(std::declval<T>()); };

// -----------------------------------------------
// -----------------------------------------------

template <activation_function_type AF>
std::ostream& operator<<(std::ostream& os, const AF& af)
{
    os << '\n' << af.repr();
    return os;
}
} // namespace matrix_activation_functions

#endif // ACTIVATION_FUNCTIONS
