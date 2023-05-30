#ifndef ACTIVATION_FUNCTIONS
#define ACTIVATION_FUNCTIONS
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

//--------------------------------------------------------------------------------------//


namespace matrix_activation_functions
{
template <typename T>
struct default_activation_function_parameters
{
    template <typename Fn>
    void mutate(Fn&&)
    {
    }

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<T, Fn, Args...>
    constexpr void fill(Fn&&, Args&&...)
    {
    }

    [[nodiscard]] constexpr auto repr() const -> const char*
    {
        return "";
    }

    [[nodiscard]] static constexpr size_t parameter_count()
    {
        return 0;
    }

    template <typename R>
    [[nodiscard]] inline static constexpr R
    L1_distance(const default_activation_function_parameters&, const default_activation_function_parameters&)
    {
        return 0;
    }

    void store(std::ofstream&) const
    {
    }

    void load(std::ifstream&)
    {
    }
};

// ReSharper disable once CppInconsistentNaming
template <typename Mat>
struct ReLU
{
    using T               = typename Mat::value_type;
    using parameters_type = default_activation_function_parameters<T>;

    inline void operator()(Mat& mat, const parameters_type&) const
    {
        ReLU_impl(mat);
    }

    inline static void ReLU_impl(Mat& mat)
    {
        mat.transform([](T x) { return std::max(T(), x); }); //  { return x > T{} ? x : T{}; });
    }

    static constexpr auto name = "ReLU";
};

// ReSharper disable once CppInconsistentNaming
template <typename Mat>
struct Sigmoid
{
    using T               = typename Mat::value_type;
    using parameters_type = default_activation_function_parameters<T>;

    inline void operator()(Mat& mat, const parameters_type&) const
    {
        Sigmoid_impl(mat);
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
    using T               = typename Mat::value_type;
    using parameters_type = default_activation_function_parameters<T>;

    inline void operator()(Mat& mat, const parameters_type&) const
    {
        Identity_impl(mat);
    }

    inline static void Identity_impl(Mat&)
    {
    }

    static constexpr auto name = "Identity";
};

template <typename Mat>
struct Tanh
{
    using T               = typename Mat::value_type;
    using parameters_type = default_activation_function_parameters<T>;

    inline void operator()(Mat& mat, const parameters_type&) const
    {
        Tanh_impl(mat);
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
    using T               = typename Mat::value_type;
    using parameters_type = default_activation_function_parameters<T>;

    inline void operator()(Mat& mat, const parameters_type&) const
    {
        GELU_impl(mat);
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
    using T               = typename Mat::value_type;
    using parameters_type = default_activation_function_parameters<T>;

    inline void operator()(Mat& mat, const parameters_type&) const
    {
        SiLU_impl(mat);
    }

    inline static void SiLU_impl(Mat& mat)
    {
        mat.transform([](T x) { return static_cast<T>(x / (T(1) + std::exp(-x))); });
    }

    static constexpr auto name = "SiLU";
};

template <typename Mat>
struct Softmax
{
    using T               = typename Mat::value_type;
    using parameters_type = default_activation_function_parameters<T>;
    using matrix_iterator = typename Mat::iterator;

    static constexpr auto N = Mat::Size_x;
    static constexpr auto M = Mat::Size_y;

    inline void operator()(Mat& mat, const parameters_type&) const
    {
        Softmax_impl(mat);
    }

    inline static void Softmax_impl(Mat& mat)
    {
        for (size_t j = 0; j != M; ++j)
        {
            matrix_iterator start   = mat.begin() + j * N;
            matrix_iterator end     = start + N;
            const auto      row_max = *std::max_element(start, end);
            T               row_sum = T();

            for (auto p = start; p != end; ++p)
            {
                *p = std::exp(*p - row_max);
                row_sum += *p;
            }

            for (; start != end; ++start)
            {
                *start /= row_sum;
            }
        }
    }

    static constexpr auto name = "Softmax";
};

/**
 * \brief Swish linear unit activation function
 */
template <typename Mat>
struct Swish
{
    using T = typename Mat::value_type;

    struct parameters_type
    {
        static constexpr T lower_range_bound = -2;
        static constexpr T upper_range_bound = 2;

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
            beta = restrict_value(fn(beta));
        }

        template <typename Fn, typename... Args>
            requires std::is_invocable_r_v<T, Fn, Args...>
        constexpr void fill(Fn&& fn, Args&&... args)
        {
            beta = restrict_value(std::invoke(fn, std::forward<Args>(args)...));
        }

        static T restrict_value(const T value)
        {
            return std::min(std::max(value, lower_range_bound), upper_range_bound);
        }

        template <typename R>
        [[nodiscard]] inline static constexpr R L1_distance(const parameters_type& p1, const parameters_type& p2)
        {
            return std::abs(p1.beta - p2.beta);
        }

        void store(std::ofstream& out) const
        {
            out << beta << "\n\n";
        }

        void load(std::ifstream& in)
        {
            in >> beta;
        }
    };

    inline void operator()(Mat& mat, const parameters_type& params) const
    {
        Swish_impl(mat, params.beta);
    }

    inline static void Swish_impl(Mat& mat, T beta)
    {
        mat.transform([_beta = beta](T x) { return static_cast<T>(x / (T(1) + std::exp(-_beta * x))); });
    }

    static constexpr auto name = "Swish";
};

template <typename Mat>
struct PReLU
{
    using T = typename Mat::value_type;

    struct parameters_type
    {
        static constexpr T lower_range_bound = -0.5;
        static constexpr T upper_range_bound = 0.5;

        T alpha;

        [[nodiscard]] static constexpr size_t parameter_count()
        {
            return 1;
        }

        [[nodiscard]] constexpr auto repr() const -> std::string
        {
            return std::string("Alpha: ") + std::to_string(alpha);
        }

        template <typename Fn>
        void mutate(Fn&& fn)
        {
            alpha = restrict_value(fn(alpha));
        }

        template <typename Fn, typename... Args>
            requires std::is_invocable_r_v<T, Fn, Args...>
        constexpr void fill(Fn&& fn, Args&&... args)
        {
            alpha = restrict_value(std::invoke(fn, std::forward<Args>(args)...));
        }

        [[nodiscard]] inline static T restrict_value(const T value)
        {
            return std::min(std::max(value, lower_range_bound), upper_range_bound);
        }

        template <typename R>
        [[nodiscard]] inline static constexpr R L1_distance(const parameters_type& p1, const parameters_type& p2)
        {
            return std::abs(p1.alpha - p2.alpha);
        }

        void store(std::ofstream& out) const
        {
            out << alpha << "\n\n";
        }

        void load(std::ifstream& in)
        {
            in >> alpha;
        }
    };

    inline void operator()(Mat& mat, const parameters_type& params) const
    {
        PReLU_impl(mat, params.alpha);
    }

    inline static void PReLU_impl(Mat& mat, T alpha)
    {
        mat.transform([_alpha = alpha](T x) {
            return static_cast<T>(std::max<T>(T(), x) + _alpha * std::min<T>(T{}, x));
        });
    }

    static constexpr auto name = "PReLU";
};

template <typename Mat>
struct Threshold
{
    using T = typename Mat::value_type;

    struct parameters_type
    {
        static constexpr T lower_threshold_bound = -10;
        static constexpr T upper_threshold_bound = 10;

        T threshold;

        [[nodiscard]] static constexpr size_t parameter_count()
        {
            return 1;
        }

        [[nodiscard]] constexpr auto repr() const -> std::string
        {
            return std::string("Threshold: ") + std::to_string(threshold);
        }

        template <typename Fn>
        void mutate(Fn&& fn)
        {
            threshold = restrict_value(fn(threshold));
        }

        template <typename Fn, typename... Args>
            requires std::is_invocable_r_v<T, Fn, Args&&...>
        constexpr void fill(Fn&& fn, Args&&... args)
        {
            threshold = restrict_value(std::invoke(fn, std::forward<Args>(args)...));
        }

        [[nodiscard]] inline static T restrict_value(const T value)
        {
            return std::min(std::max(value, lower_threshold_bound), upper_threshold_bound);
        }

        template <typename R>
        [[nodiscard]] inline static constexpr R L1_distance(const parameters_type& p1, const parameters_type& p2)
        {
            return std::abs(p1.threshold - p2.threshold);
        }

        void store(std::ofstream& out) const
        {
            out << threshold << "\n\n";
        }

        void load(std::ifstream& in)
        {
            in >> threshold;
        }
    };

    inline void operator()(Mat& mat, const parameters_type& params) const
    {
        Threshold_impl(mat, params.threshold);
    }

    inline static void Threshold_impl(Mat& mat, T threshold)
    {
        // mat.transform([_threshold = threshold](T x) { return x > _threshold ? T(1) : T(); });
        mat.transform([_threshold = threshold](T x) { return x > T() ? T(1) : T(); });
    }

    static constexpr auto name = "Threshold";
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
        Softmax,
        Swish,
        PReLU,
        Threshold
    };
};

template <typename Mat, matrix_activation_functions::Identifiers::Identifiers_ Function_Identifier>
// ReSharper disable once CppNotAllPathsReturnValue
[[nodiscard]] constexpr auto choose_func() noexcept
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
    else if constexpr (Function_Identifier == matrix_activation_functions::Identifiers::Softmax)
        return matrix_activation_functions::Softmax<Mat>();
    else if constexpr (Function_Identifier == matrix_activation_functions::Identifiers::Swish)
        return matrix_activation_functions::Swish<Mat>();
    else if constexpr (Function_Identifier == matrix_activation_functions::Identifiers::PReLU)
        return matrix_activation_functions::PReLU<Mat>();
    else if constexpr (Function_Identifier == matrix_activation_functions::Identifiers::Threshold)
        return matrix_activation_functions::Threshold<Mat>();
    std::unreachable();
}

template <typename Mat, matrix_activation_functions::Identifiers::Identifiers_ Function_Identifier>
class activation_function
{
public:
    inline static constexpr auto activation_function_impl = choose_func<Mat, Function_Identifier>();

    using activation_function_type = decltype(activation_function_impl);
    using parameters_type          = typename activation_function_type::parameters_type;

    parameters_type params;

    void operator()(Mat& mat) const
    {
        std::invoke(activation_function_impl, mat, params);
    }

    template <typename Fn>
    void mutate_params(Fn&& fn)
    {
        params.mutate(fn);
    }

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<typename activation_function_type::T, Fn, Args...>
    constexpr void fill(Fn&& fn, Args&&... args)
    {
        params.fill(fn, std::forward<Args>(args)...);
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

    template <typename R>
    [[nodiscard]] static R L1_distance(const activation_function& af1, const activation_function& af2)
    {
        return parameters_type::template L1_distance<R>(af1.params, af2.params);
    }

    void store(std::ofstream& out) const
    {
        params.store(out);
    }

    void load(std::ifstream& in)
    {
        params.load(in);
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
    os << af.repr();
    return os;
}
} // namespace matrix_activation_functions

#endif // ACTIVATION_FUNCTIONS
