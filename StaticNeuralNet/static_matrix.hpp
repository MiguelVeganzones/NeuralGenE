#pragma once

#ifndef STATIC_MATRIX
#define STATIC_MATRIX

#include "Random.hpp"
#include "Stopwatch.hpp"
#include "cx_helper_functions.hpp"
#include <array>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstring> //memcpy
#include <fstream>
#include <functional>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <type_traits>


/*
based on:
  msvc implementation of std::array
  and: https://github.com/douglasrizzo/matrix/blob/master/include/nr3/nr3.h
*/

#ifndef NDEBUG
#define CHECKBOUNDS
#endif

/*
#ifdef _CHECKBOUNDS_
if (i < 0 || i >= nn) {
  throw("NRmatrix subscript out of bounds");
}
#endif
*/

/*
Basic dense matrix library

Row major, dense, and efficient static matrix implementation.

It is highly based in std::array, but with two dimensional indexing.
It is a trivial type, and has standar layout (called POD type (Pre c++20)).
That means it is trivially copiable, trivially copy constructible, and others.

Most functionality is constexpr and can be exceuted in compile time. This is used
for its unit testing and can be a useful feature in some situations.

Added functionality is mainly math operations, to make this a matrix and not just a 2D array,
convenience functions, and basic functionality that can be useful in GA algorithms.

This is not a matrix intended to do math, but does have math functionality. Which means that
algorithms like inverse or RREF might yield wrong results when matrices are not invertible or
is their determinant is too small (Note: having a small determinant is not a condition for closeness
to singularity, but can be a good heuristic). This is specially the case when using single precision
floating point numbers, and thus, double recision should be used for math applications. Results of
this operations should be double checked if singular matrices might appear as the algorithms do not
cover corner cases.

Code bloat could be an issue if using multiple types of matrices (shapes and value types). This has
not been taken into account during design.

I decided to use - T const& - in most cases because i found it easier to read due to long names

*/

namespace ga_sm
{

template <typename T, size_t Size>
class matrix_const_iterator
{
public:
    using value_type = T;
    using pointer    = const T*;
    using reference  = const T&;

    constexpr matrix_const_iterator() noexcept :
        m_Ptr(nullptr)
    {
    }

    constexpr explicit matrix_const_iterator(pointer p_arg, size_t offs = 0) noexcept :
        m_Ptr(p_arg + offs)
    {
    }

    [[nodiscard]] constexpr reference operator*() const noexcept
    {
        return *m_Ptr;
    }

    [[nodiscard]] constexpr pointer operator->() const noexcept
    {
        return m_Ptr;
    }

    //++iter; retuns by reference mutated original object. Idem for the rest
    constexpr matrix_const_iterator& operator++() noexcept
    {
        ++m_Ptr;
        return *this;
    }

    // iter++; returns by value copy of the original iterator. Idem for the rest
    constexpr matrix_const_iterator operator++(int) noexcept
    {
        matrix_const_iterator Tmp = *this;
        ++m_Ptr;
        return Tmp;
    }

    //--iter;
    constexpr matrix_const_iterator& operator--() noexcept
    {
        --m_Ptr;
        return *this;
    }

    // iter--;
    constexpr matrix_const_iterator operator--(int) noexcept
    {
        matrix_const_iterator Tmp = *this;
        --m_Ptr;
        return Tmp;
    }

    constexpr matrix_const_iterator& operator+=(const ptrdiff_t offs) noexcept
    {
        m_Ptr += offs;
        return *this;
    }

    [[nodiscard]] constexpr matrix_const_iterator operator+(const ptrdiff_t offs) const noexcept
    {
        matrix_const_iterator Tmp = *this;
        Tmp += offs;
        return Tmp;
    }

    constexpr matrix_const_iterator& operator-=(const ptrdiff_t offs) noexcept
    {
        m_Ptr -= offs;
        return *this;
    }

    [[nodiscard]] constexpr matrix_const_iterator operator-(const ptrdiff_t offs) const noexcept
    {
        matrix_const_iterator Tmp = *this;
        Tmp -= offs;
        return Tmp;
    }

    [[nodiscard]] constexpr ptrdiff_t operator-(const matrix_const_iterator& Rhs) const noexcept
    {
        return m_Ptr - Rhs.m_Ptr;
    }

    [[nodiscard]] constexpr reference operator[](const ptrdiff_t offs) const noexcept
    {
        return m_Ptr[offs];
    }

    [[nodiscard]] constexpr bool operator==(const matrix_const_iterator& Rhs) const noexcept
    {
        return m_Ptr == Rhs.m_Ptr;
    }

    [[nodiscard]] constexpr bool operator!=(const matrix_const_iterator& Rhs) const noexcept
    {
        return !(*this == Rhs);
    }

    [[nodiscard]] constexpr bool operator>(const matrix_const_iterator& Rhs) const noexcept
    {
        return m_Ptr > Rhs.m_Ptr;
    }

    [[nodiscard]] constexpr bool operator<(const matrix_const_iterator& Rhs) const noexcept
    {
        return m_Ptr < Rhs.m_Ptr;
    }

    constexpr void seek_to(pointer it) noexcept
    {
        m_Ptr = it;
    }

    [[nodiscard]] pointer unwrapped() const noexcept
    {
        return m_Ptr;
    }

private:
    pointer m_Ptr;
};

//-------------------------------------------------------------------//

template <typename T, size_t Size>
class matrix_iterator : public matrix_const_iterator<T, Size>
{
public:
    using My_Base    = matrix_const_iterator<T, Size>;
    using value_type = T;
    using pointer    = T*;
    using reference  = T&;

    constexpr matrix_iterator() noexcept = default;

    constexpr explicit matrix_iterator(pointer p_arg, size_t offs = 0) noexcept :
        My_Base(p_arg, offs)
    {
    }

    [[nodiscard]] constexpr reference operator*() const noexcept
    {
        return const_cast<reference>(My_Base::operator*());
    }

    [[nodiscard]] constexpr pointer operator->() const noexcept
    {
        return const_cast<pointer>(My_Base::operator->());
    }

    constexpr matrix_iterator& operator++() noexcept
    {
        My_Base::operator++();
        return *this;
    }

    constexpr matrix_iterator operator++(int) noexcept
    {
        matrix_iterator tmp = *this;
        My_Base::operator++();
        return tmp;
    }

    constexpr matrix_iterator& operator--() noexcept
    {
        My_Base::operator--();
        return *this;
    }

    constexpr matrix_iterator operator--(int) noexcept
    {
        matrix_iterator tmp = *this;
        My_Base::operator--();
        return tmp;
    }

    constexpr matrix_iterator& operator+=(const ptrdiff_t offs) noexcept
    {
        My_Base::operator+=(offs);
        return *this;
    }

    [[nodiscard]] constexpr matrix_iterator operator+(const ptrdiff_t offs) const noexcept
    {
        matrix_iterator tmp = *this;
        tmp += offs;
        return tmp;
    }

    constexpr matrix_iterator& operator-=(const ptrdiff_t offs) noexcept
    {
        My_Base::operator-=(offs);
        return *this;
    }

    [[nodiscard]] constexpr matrix_iterator operator-(const ptrdiff_t offs) const noexcept
    {
        matrix_iterator tmp = *this;
        tmp -= offs;
        return tmp;
    }

    [[nodiscard]] constexpr reference operator[](const ptrdiff_t offs) const noexcept
    {
        return const_cast<reference>(My_Base::operator[](offs));
    }

    [[nodiscard]] constexpr pointer unwrapped() const noexcept
    {
        return const_cast<pointer>(My_Base::unwrapped());
    }
};

//-------------------------------------------------------------------//

//-------------------------------------------------------------------//
template <typename T, size_t M, size_t N>
    requires(std::is_arithmetic_v<T> && (N > 0) && (M > 0))
class static_matrix
{
public:
    using value_type      = T;
    using size_type       = size_t;
    using pointer         = T*;
    using const_pointer   = const T*;
    using reference       = T&;
    using const_reference = const T&;
    // using iterator        = pointer;
    // using const_iterator  = const_pointer;
    using iterator       = matrix_iterator<T, M * N>;
    using const_iterator = matrix_const_iterator<T, M * N>;
    using row_matrix     = static_matrix<T, 1, N>;
    using column_matrix  = static_matrix<T, M, 1>;

    inline static constexpr auto Size_y = M;
    inline static constexpr auto Size_x = N;
    inline static constexpr auto Size   = N * M;

    T m_Elems[M * N];

    /*--------------------------------------------*/

    [[nodiscard]] constexpr size_type size() const noexcept
    {
        return M * N;
    }

    constexpr void fill(const T& val)
    {
        fill_n(iterator(m_Elems), M * N, val);
    }

    template <class Fn, class... Args>
        requires std::is_invocable_r_v<T, Fn, Args...>
    void fill(Fn&& fn, Args&&... args) noexcept(noexcept(std::invoke(fn, std::forward<Args>(args)...)))
    {
        for (iterator it = begin(); it != end(); ++it)
            *it = static_cast<T>(std::invoke(fn, std::forward<Args>(args)...));
    }

    constexpr void fill_n(iterator dest, const ptrdiff_t count, const T val) noexcept
    {
#ifdef _CHECKBOUNDS_
        const bool indexing_b = *dest < *m_Elems; // pointer before array
        const bool indexing_p = *dest + count > *m_Elems + M * N; // write past array limits
        if (indexing_b || indexing_p)
        {
            std::cout << "Indexing before array: " << indexing_b << "Indexing past array: " << indexing_p << std::endl;
            std::cout << "fill_n subscript out of bounds\n";
            std::exit(EXIT_FAILURE);
        }
#endif

        for (ptrdiff_t i = 0; i != count; ++i, ++dest)
        {
            *dest = val;
        }
    }

    /// <summary>
    /// Fills the matrix with ascending values from 0 to M*N-1
    /// </summary>
    constexpr void iota_fill()
    {
        std::iota(begin(), end(), T(0));
    }

    template <typename Fn>
        requires std::is_invocable_r_v<T, Fn, T>
    constexpr void transform(Fn&& fn)
    {
        for (auto& e : *this)
            e = fn(e);
    }

    template <typename Fn>
        requires std::is_invocable_r_v<T, Fn, T>
    [[nodiscard]] static_matrix apply_fn(Fn&& fn) const noexcept(noexcept(fn))
    {
        static_matrix ret{};
        std::transform(begin(), end(), ret.begin(), std::forward<Fn>(fn));
        return ret;
    }

    // TODO Deducing this (not yet supported)
    //  [[nodiscard]] constexpr auto&& begin(this auto&&) const noexcept
    //  {
    //      return m_Elems;
    //  }

    // [[nodiscard]] constexpr auto&& end(this auto&& self) noexcept
    // {
    //     return m_Elems + M * N;
    // }

    [[nodiscard]] constexpr iterator begin() noexcept
    {
        return iterator(m_Elems);
    }

    [[nodiscard]] constexpr const_iterator begin() const noexcept
    {
        return const_iterator(m_Elems);
    }

    [[nodiscard]] constexpr const_iterator cbegin() const noexcept
    {
        return const_iterator(m_Elems);
    }

    [[nodiscard]] constexpr iterator end() noexcept
    {
        return iterator(m_Elems, M * N);
        // return m_Elems + M * N;
    }

    [[nodiscard]] constexpr const_iterator end() const noexcept
    {
        return const_iterator(m_Elems, M * N);
        // return m_Elems + M * N;
    }

    [[nodiscard]] constexpr const_iterator cend() const noexcept
    {
        return const_iterator(m_Elems, M * N);
        // return m_Elems + M * N;
    }

    [[nodiscard]] inline constexpr reference operator[](const size_t j, const size_t i) noexcept
    {
        assert(j < M and i < N);
        return m_Elems[j * N + i];
    }

    [[nodiscard]] inline constexpr const_reference operator[](const size_t j, const size_t i) const noexcept
    {
        assert(j < M and i < N);
        return m_Elems[j * N + i];
    }

    [[nodiscard]] inline constexpr reference operator[](const size_t j) noexcept
        requires(M == 1 || N == 1)
    {
        assert(j < M * N);
        return m_Elems[j];
    }

    [[nodiscard]] inline constexpr const_reference operator[](const size_t j) const noexcept
        requires(M == 1 || N == 1)
    {
        assert(j < M * N);
        return m_Elems[j];
    }

    [[nodiscard]] constexpr static_matrix operator+(static_matrix const& other) const
    {
        static_matrix ret(*this);
        for (size_t i = 0; i != M * N; ++i)
        {
            ret.m_Elems[i] += other.m_Elems[i];
        }
        return ret;
    }

    [[nodiscard]] constexpr static_matrix operator-(static_matrix const& other) const
    {
        static_matrix ret(*this);
        for (size_t i = 0; i != M * N; ++i)
        {
            ret.m_Elems[i] -= other.m_Elems[i];
        }
        return ret;
    }

    [[nodiscard]] constexpr static_matrix operator*(const T p) const
    {
        static_matrix ret(*this);
        for (auto& e : ret)
            e *= p;
        return ret;
    }

    constexpr static_matrix operator*=(const T p) noexcept
    {
        for (auto& e : *this)
            e *= p;
        return *this;
    }

    [[nodiscard]] constexpr static_matrix operator/(const T p) const
    {
        static_matrix ret(*this);
        for (auto& e : ret)
            e /= p;
        return ret;
    }

    constexpr static_matrix operator/=(const T p)
    {
        assert(p != 0);
        for (auto& e : *this)
            e /= p;
        return *this;
    }

    constexpr static_matrix operator-=(const T p)
    {
        for (auto& e : *this)
            e -= p;
        return *this;
    }

    /*-----------------------------------------------------------------
              ###		Utility		###
    ------------------------------------------------------------------*/

    template <typename R = T>
    [[nodiscard]] constexpr R sum() const noexcept
    {
        return std::accumulate(begin(), end(), R{ 0 });
    }

    template <std::floating_point F = float>
    [[nodiscard]] constexpr F mean() const noexcept
    {
        return sum<F>() / F(this->size());
    }

    /**
     * \brief Rescales the L1 dist_op of the matrix to the input dist_op value by dividing every element by a constant
     * amount.
     *
     * \param dist_op The resulting L1 dist_op of the matrix
     *
     * \note Only works for floating point matrix value types
     * \throws std::logic_error (or other, implementation dependent) when divsion by zero
     */
    constexpr void rescale_L_1_1_norm(const T dist_op = T(1)) noexcept(false)
        requires std::is_floating_point_v<T>
    {
        assert(dist_op != 0.f);
        const auto sum    = sum();
        const auto factor = static_cast<T>(dist_op / sum);
        for (auto& e : *this)
            e *= factor;
    }

    /// <summary>
    /// Rescales all values in the matrix to fit a normal distribution N(0,1) (mean 0 and sttdev 1)
    /// </summary>
    void standarize()
        requires std::is_floating_point_v<T>
    {
        const auto mean = this->mean();

        T s = T{ 0 };
        for (auto& e : *this)
        {
            e -= mean;
            s += e * e;
        }
        const T stddev = std::sqrt(s / T(this->size()));
        for (auto& e : *this)
            e /= stddev;
    }

    // Ofstream file must be closed outside of this function
    void store(std::ofstream& out) const
    {
        for (size_t j = 0; j != M; ++j)
        {
            for (size_t i = 0; i != N; ++i)
            {
                out << this->operator[](j, i) << ' ';
            }
            out << '\n';
        }
        out << '\n';
    }

    // Overrides matrix with matrix read from in
    // Ifstream file should be closed outside this function
    void load(std::ifstream& in)
    {
        for (auto& e : m_Elems)
            in >> e;
    }
};

// -----------------------------------------------
// Static matrix concept
// -----------------------------------------------

template <typename T, size_t M, size_t N>
void matrix_dummy(static_matrix<T, M, N>)
{
}

template <typename T>
concept static_matrix_type = requires { matrix_dummy(std::declval<T&>()); };

// -----------------------------------------------
// -----------------------------------------------

template <typename R, static_matrix_type Matrix>
[[nodiscard]] constexpr auto cast_to(Matrix const& src) noexcept -> static_matrix<R, Matrix::Size_y, Matrix::Size_x>
{
    constexpr auto M = Matrix::Size_y;
    constexpr auto N = Matrix::Size_x;

    static_matrix<R, M, N> ret{};
    for (size_t j = 0; j != M; ++j)
    {
        for (size_t i = 0; i != N; ++i)
        {
            ret[j, i] = static_cast<R>(src[j, i]);
        }
    }
    return ret;
}

template <size_t Out_M, size_t Out_N, static_matrix_type Matrix>
    requires((Matrix::Size_y * Matrix::Size_x) == (Out_M * Out_N))
[[nodiscard]] constexpr auto cast_to_shape(Matrix const& src) noexcept
    -> static_matrix<typename Matrix::value_type, Out_M, Out_N>
{
    constexpr auto M = Matrix::Size_y;
    constexpr auto N = Matrix::Size_x;
    using T          = typename Matrix::value_type;

    static_matrix<T, Out_M, Out_N> ret{};
    std::memcpy(ret.m_Elems, src.m_Elems, N * M * sizeof(T));
    return ret;
}

template <static_matrix_type Matrix>
std::ostream& operator<<(std::ostream& os, Matrix const& mat)
{
    if (!std::is_integral_v<typename Matrix::value_type>)
    {
        os << std::fixed;
        os << std::setprecision(4);
    }
    std::cout << "[";
    for (size_t j = 0; j != Matrix::Size_y; ++j)
    {
        std::cout << "\n [ ";
        for (size_t i = 0; i != Matrix::Size_x; ++i)
        {
            os << mat[j, i] << ' ';
        }
        os << "]";
    }
    std::cout << "\n]\n";
    os << std::defaultfloat;
    return os;
}

/**
 * \brief Returns exactly equals for integral types and nearly equals if else.
 *        Default tolerance used is 1e-4 for non integral types
 */
template <static_matrix_type Matrix>
[[nodiscard]] constexpr auto operator==(Matrix const& mat1, Matrix const& mat2) -> bool
{
    if constexpr (std::is_floating_point_v<typename Matrix::value_type>)
        return nearly_equals(mat1, mat2);
    else
        return exactly_equals(mat1, mat2);
}

template <static_matrix_type Matrix>
[[nodiscard]] constexpr bool operator!=(Matrix const& mat1, Matrix const& mat2)
{
    return !operator==(mat1, mat2);
}

template <static_matrix_type Matrix>
[[nodiscard]] constexpr bool nearly_equals(
    Matrix const& mat1, Matrix const& mat2, const typename Matrix::value_type epsilon = 1e-4
) noexcept
{
    for (size_t j = 0; j != Matrix::Size_y; ++j)
    {
        for (size_t i = 0; i != Matrix::Size_x; ++i)
        {
            const auto d = mat1[j, i] - mat2[j, i];
            if (cx_helper_func::cx_abs(d) > epsilon)
                return false;
        }
    }
    return true;
}

template <static_matrix_type Matrix>
[[nodiscard]] constexpr bool exactly_equals(Matrix const& mat1, Matrix const& mat2) noexcept
{
    for (size_t j = 0; j != Matrix::M; ++j)
    {
        for (size_t i = 0; i != Matrix::N; ++i)
        {
            if (mat1[j, i] != mat2[j, i])
                return false;
        }
    }
    return true;
}

//-----------------------------------------------------------------------------
//------------ GA-Specific functionallity -------------------------------------
//-----------------------------------------------------------------------------

template <static_matrix_type Matrix>
void in_place_x_crossover(Matrix& mat1, Matrix& mat2) noexcept
{
    constexpr auto M = Matrix::Size_y;
    constexpr auto N = Matrix::Size_x;

    // indices to slice. a: horizontal, b: vertical
    const size_t a = random::randsize_t(0, M);
    const size_t b = random::randsize_t(0, N);

    // return swapped original matrices to avoid irrelevant operations
    if ((a == 0 or a == M) and (b == 0 or b == N))
    {
        return;
    }

    // minimize swaps -> less area
    const size_t area1 = a * b + (M - a) * (N - b);
    const size_t area2 = M * N - area1;

    const bool d = area1 < area2; // block diagonal to swap. True for main diagonal, false for anti diagonal

    // swap first block
    for (size_t j = 0; j != a; ++j)
    {
        for (size_t i = (d ? 0 : b); i != (d ? b : N); ++i)
        {
            std::swap(mat1[j, i], mat2[j, i]);
        }
    }

    // swap second block
    for (size_t j = a; j != M; ++j)
    {
        for (size_t i = (d ? b : 0); i != (d ? N : b); ++i)
        {
            std::swap(mat1[j, i], mat2[j, i]);
        }
    }
}

template <static_matrix_type Matrix>
void to_target_x_crossover(Matrix const& in_mat1, Matrix const& in_mat2, Matrix& out_mat1, Matrix& out_mat2) noexcept
{
    // indices to slice. a: horizontal, b: vertical
    const auto a = static_cast<size_t>(random::randint(0, Matrix::Size_y));
    const auto b = static_cast<size_t>(random::randint(0, Matrix::Size_x));

    for (size_t j = 0; j != Matrix::Size_y; ++j)
    {
        for (size_t i = 0; i != Matrix::Size_x; ++i)
        {
            auto main_diagonal = (j < a) == (i < b);
            out_mat1[j, i]     = main_diagonal ? in_mat1[j, i] : in_mat2[j, i];
            out_mat2[j, i]     = main_diagonal ? in_mat2[j, i] : in_mat1[j, i];
        }
    }
}

template <static_matrix_type Matrix>
[[nodiscard]] std::pair<Matrix, Matrix> x_crossover(Matrix const& mat1, Matrix const& mat2) noexcept
{
    // setup return matrices.
    Matrix ret1(mat1);
    Matrix ret2(mat2);

    in_place_x_crossover(ret1, ret2);

    return std::make_pair(ret1, ret2);
}

//-----------------------------------------------------------------------------
//------------ Maths utility --------------------------------------------------
//-----------------------------------------------------------------------------

template <typename T, size_t M, size_t K, size_t N>
[[nodiscard]] constexpr static_matrix<T, M, N> matrix_mul(
    static_matrix<T, M, K> const& mat1, static_matrix<T, K, N> const& mat2
) noexcept
{
    static_matrix<T, M, N> ret{};

    for (size_t j = 0; j != M; ++j)
    {
        for (size_t k = 0; k != K; ++k)
        {
            for (size_t i = 0; i != N; ++i)
            {
                ret[j, i] += mat1[j, k] * mat2[k, i];
            }
        }
    }
    return ret;
}

// // TODO check impl
// template <size_t M, size_t K, size_t N>
// [[nodiscard]] constexpr static_matrix<float, M, N> matrix_mul_avx(
//     static_matrix<float, M, K> const& mat1, static_matrix<float, K, N> const& mat2
// ) noexcept
// {
//     static_matrix<float, M, N> ret{};
//     static_matrix<float, N, K> tmat2 = transpose(mat2);

//     if constexpr (K > 7)
//     {
//         for (size_t j = 0; j != M; ++j)
//         {
//             for (size_t jj = 0; jj < K - 7; jj += 8)
//             {
//                 // std::cout << "Here2\n";
//                 const __m256 row_piece = _mm256_loadu_ps(mat1.m_Elems + j * K + jj);
//                 for (size_t k = 0; k != N; ++k)
//                 {
//                     // std::cout << "Here3\n";
//                     const __m256 v_piece = _mm256_loadu_ps(tmat2.m_Elems + k * K + jj);

//                     const __m256 m256_dot_piece = _mm256_dp_ps(v_piece, row_piece, 0xf1); // compare 0xff with 0xf1

//                     const auto float8_dot_piece = std::bit_cast<std::array<float, 8>>(m256_dot_piece);
//                     ret[j, k] += float8_dot_piece[0] + float8_dot_piece[4]; // Now without UB! std::bit_cast edition
//                 }
//             }
//         }
//     }
//     if constexpr (K % 8 != 0)
//     {
//         for (size_t j = 0; j != M; ++j)
//         {
//             for (size_t k = (K & (~static_cast<size_t>(7))); k < K; ++k)
//             {
//                 for (size_t i = 0; i != N; ++i)
//                 {
//                     ret[j, i] += mat1[j, k] * mat2[k, i];
//                 }
//             }
//         }
//     }
//     return ret;
// }

template <typename T, size_t M, size_t N>
[[nodiscard]] constexpr static_matrix<T, M, N> matrix_vec_add(
    static_matrix<T, M, N> const& mat, static_matrix<T, 1, N> const& vec
) noexcept
{
    if constexpr (M == 1)
        return mat + vec;
    else
    {
        auto ret{ mat };
        for (size_t i = 0; i != N; ++i)
        {
            for (size_t j = 0; j != M; ++j)
            {
                ret[j, i] += vec[0, i];
            }
        }
        return ret;
    }
}

/**
 * \brief
 *    AVX matrix multiplication
 *    Speed up over regular matrix mul for 40x40 matrices and above
 * \tparam M Matrix dimension 1
 * \tparam N Matrix dimension 2
 * \param mat Matrix to multiply
 * \param vec Vector to multiply
 * \return Multiplied matrix
 */
template <size_t M, size_t N>
[[nodiscard]] static_matrix<float, M, 1> matrix_vector_mul_float_avx(
    static_matrix<float, M, N> const& mat, static_matrix<float, N, 1> const& vec
)
{
    static_matrix<float, M, 1> ret{};
    if constexpr (N > 7)
    {
        for (size_t i = 0; i < N - 7; i += 8)
        {
            const __m256 v_piece = _mm256_loadu_ps(vec.m_Elems + i);
            for (size_t j = 0; j != M; ++j)
            {
                const __m256 row_piece      = _mm256_loadu_ps(mat.m_Elems + i + j * N);
                const __m256 m256_dot_piece = _mm256_dp_ps(v_piece, row_piece, 0xf1); // compare 0xff with 0xf1

                const auto float8_dot_piece = std::bit_cast<std::array<float, 8>>(m256_dot_piece);
                ret[j, 0] += float8_dot_piece[0] + float8_dot_piece[4]; // Now without UB! std::bit_cast edition
            }
        }
    }
    // last iterations (lower 3 bits of N)
    if constexpr (N % 8 != 0)
    {
        constexpr size_t iterations_completed = (N & (~static_cast<size_t>(7)));
        for (size_t ii = iterations_completed; ii != N; ++ii)
        {
            for (size_t j = 0; j != M; ++j)
            {
                ret[j, 0] += mat[j, ii] * vec[ii, 0];
            }
        }
    }
    return ret;
}

// multiplies every column of a matrix by the element of a row vector of same index as the column
template <typename T, size_t M, size_t N>
[[nodiscard]] constexpr static_matrix<T, M, N> vector_expand(
    static_matrix<T, 1, N> const& row_vector, static_matrix<T, M, N> const& col_vectors
)
{
    auto ret{ col_vectors };
    for (size_t j = 0; j != M; ++j)
    {
        for (size_t i = 0; i != N; ++i)
        {
            ret[j, i] *= row_vector[0, i];
        }
    }
    return ret;
}

/*
Multiplies pairs of vectors stored in two matrices
Useful to make multiple vector multiplications with one operation and one data structure
*/
template <typename T, size_t M, size_t N>
[[nodiscard]] constexpr static_matrix<T, 1, M> multiple_dot_product(
    static_matrix<T, M, N> const& row_vectors, static_matrix<T, N, M> const& col_vectors
)
{
    static_matrix<T, 1, M> ret{};
    for (size_t j = 0; j != M; ++j)
    {
        for (size_t i = 0; i != N; ++i)
        {
            ret[0, j] += row_vectors[j, i] * col_vectors[i, j];
        }
    }
    return ret;
}

template <typename T, size_t M, size_t N, size_t P, typename Fn>
    requires std::is_invocable_r_v<T, Fn, T>
[[nodiscard]] constexpr static_matrix<T, M, 1> multiply_add_activate(
    const static_matrix<T, M, N>& mat_mul_a,
    const static_matrix<T, N, P>& mat_mul_b,
    const static_matrix<T, M, P>& mat_add,
    Fn&&                          activation_func
)
{
    auto mul = matrix_mul(mat_mul_a, mat_mul_b);
    assert(mul.size() == mat_add.size());
    for (auto it = mat_add.begin(); auto& e : mul)
    {
        e = std::invoke(activation_func, (e + *(it++)));
    }
    return mul;
}

//-----------------------------------------------------------------------------
//------------ Math related functionality  -------------------------------------
//-----------------------------------------------------------------------------

/*
efficient LU decompoition of N by N matrix M
returns:
  bool: det(M) != 0 //If matrix is inversible or not
  double: det(M)
  static_matrix LU: L is lower triangular with unit diagonal (implicit)
             U is upped diagonal, including diagonal elements

*/
template <typename T_In, typename T_Out, size_t N>
    requires(N > 1)
[[nodiscard]] constexpr std::tuple<bool, double, static_matrix<T_Out, N, N>, std::array<size_t, N>> PII_LUDecomposition(
    static_matrix<T_In, N, N> const& scr
)
{
    /*
    source:
    http://web.archive.org/web/20150701223512/http://download.intel.com/design/PentiumIII/sml/24504601.pdf

    Factors "_Source" matrix into out=LU where L is lower triangular and U is upper
    triangular. The matrix is overwritten by LU with the diagonal elements
    of L (which are unity) not stored. This must be a square n x n matrix.
    */

    using cx_helper_func::cx_abs; // constexpr abs
    using return_type_congruent_matrix = static_matrix<T_Out, N, N>;

    return_type_congruent_matrix out = cast_to<T_Out>(scr);

    double det = 1.0;

    // Initialize the pointer vector.
    std::array<size_t, N> r_idx{}; // row index
    std::iota(r_idx.begin(), r_idx.end());

    // LU factorization.
    for (size_t p = 0; p != N - 1; ++p)
    {
        // Find pivot element.
        for (size_t j = p + 1; j != N; ++j)
        {
            if (cx_abs(out[r_idx[j], p]) > cx_abs(out[r_idx[p], p]))
            {
                // Swap the index for the p pivot row if necessary.;
                std::swap(r_idx[j], r_idx[p]);
                det = -det;
                // RIdx[p] now has the index of the row to consider the pth
            }
        }

        if (out(r_idx[p], p) == 0)
        {
            // The matrix is singular. //or not invertible by this method until fixed (no
            // permutations)
            return std::tuple<bool, double, return_type_congruent_matrix, std::array<size_t, N>>(
                false, 0.0, std::move(out), {}
            );
        }
        // Multiply the diagonal elements.
        det *= out[r_idx[p], p];

        // Form multiplier.
        for (size_t j = p + 1; j != N; ++j)
        {
            out[r_idx[j], p] /= out[r_idx[p], p];
            // Eliminate [p].
            for (size_t i = p + 1; i != N; ++i)
            {
                out[r_idx[j], i] -= out[r_idx[j], p] * out[r_idx[p], i];
            }
        }
    }
    det *= out[r_idx[N - 1], N - 1]; // multiply last diagonal element

    const std::array<size_t, N> ri(r_idx);

    // reorder output for simplicity
    for (size_t j = 0; j != N; ++j)
    {
        if (j != r_idx[j])
        {
            for (size_t i = 0; i != N; ++i)
            {
                std::swap(out[j, i], out[r_idx[j], i]);
            }
            std::swap(r_idx[j], r_idx[static_cast<size_t>(std::find(r_idx.begin(), r_idx.end(), j) - r_idx.begin())]);
        }
    }
    // F.21
    // https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#f21-to-return-multiple-out-values-prefer-returning-a-struct-or-tuple
    return { det != 0.0, det != 0.0 ? det : NAN, std::move(out), std::move(ri) };
}

/**
 * \brief Calculates determinant of a square matrix
 * \param scr Input square matrix
 * \return Determinant of the input matrix
 */
template <typename T, size_t N>
[[nodiscard]] constexpr double determinant(static_matrix<T, N, N> const& scr)
{
    return std::get<1>(PII_LUDecomposition(scr));
}

/**
 * \brief Mutates matrix to its reduced row echelon form
 * Can be used to solve systems of linear equations with an augmented matrix
 * or to invert a matrix M : ( M | I ) -> ( I | M^-1 )
 * \param scr Input matrix
 * \return Bool indicating whether or not the matrix is invertible and the transformation succeeded
 *
 * \note Uses Gauss-Jordan elimination with partial pivoting
 *       Mutates input
 */
template <std::floating_point T, size_t M, size_t N>
    requires(M <= N)
[[nodiscard]] constexpr bool RREF(static_matrix<T, M, N>& scr)
{
    using cx_helper_func::cx_abs; // constexpr abs

    std::array<size_t, N> r_idx{}; // row index
    std::iota(r_idx.begin(), r_idx.end(), 0);

    for (size_t p = 0; p != M; ++p)
    {
        for (size_t j = p + 1; j < M; ++j)
        {
            if (cx_abs(scr[r_idx[p], p]) < cx_abs(scr[r_idx[j], p]))
            {
                std::swap(r_idx[p], r_idx[j]);
            }
        }

        if (scr[r_idx[p], p] == 0)
            return false; // matrix is singular

        for (size_t i = p + 1; i < N; ++i)
        {
            scr[r_idx[p], i] /= scr[r_idx[p], p];
        }
        scr[r_idx[p], p] = 1;

        for (size_t j = 0; j != M; ++j)
        {
            if (j != p)
            {
                for (size_t i = p + 1; i < N; ++i)
                { // p+1 to avoid removing each rows' scale factor
                    scr(r_idx[j], i) -= scr(r_idx[p], i) * scr(r_idx[j], p);
                }
                scr(r_idx[j], p) = 0;
            }
        }
    }

    // reorder matrix
    for (size_t j = 0; j != M; ++j)
    {
        if (j != r_idx[j])
        {
            for (size_t i = 0; i != N; ++i)
            {
                std::swap(scr[j, i], scr[r_idx[j], i]);
            }
            std::swap(r_idx[j], r_idx[static_cast<size_t>(std::find(r_idx.begin(), r_idx.end(), j) - r_idx.begin())]);
        }
    }
    return true;
}

/**
 * \brief Inverts N*N matrix using gauss-jordan reduction with pivoting
 *        Not the most stable algorithm, nor the most efficient in space or time
 *        Beware of stability problems with singular or close to singularity matrices
 * \tparam T_In Value_Type of the input matrix
 * \tparam T_Out Value_Type of the output matrix
 * \tparam N Size of the input square matrix
 * \param scr Matrix to invert
 * \return Tuple containing a bool that represents whether or not the operation succeeded and the inverted matrix if the
 *         operation was successful, empty matrix otherwise
 */
template <typename T_In, std::floating_point T_Out = double, size_t N>
[[nodiscard]] constexpr std::tuple<bool, static_matrix<T_Out, N, N>> inverse(static_matrix<T_In, N, N> const& scr)
{
    // Tmp == ( M | I )
    static_matrix<T_Out, N, N * 2> tmp{};
    // M
    for (size_t j = 0; j != N; ++j)
    {
        for (size_t i = 0; i != N; ++i)
        {
            tmp[j, i] = static_cast<T_Out>(scr[j, i]);
        }
    }
    // I
    for (size_t j = 0; j != N; ++j)
    {
        tmp[j, j + N] = 1;
    }
    // tmp = ( I | M^-1 )
    const bool invertible = RREF(tmp);

    static_matrix<T_Out, N, N> inverse{};

    if (invertible)
    {
        for (size_t j = 0; j != N; ++j)
        {
            for (size_t i = 0; i != N; ++i)
            {
                inverse[j, i] = tmp[j, i + N];
            }
        }
    }

    // F.21
    // https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#f21-to-return-multiple-out-values-prefer-returning-a-struct-or-tuple
    return { invertible, std::move(inverse) };
}

template <typename T, size_t N>
[[nodiscard]] constexpr static_matrix<T, N, N> identity_matrix()
{
    static_matrix<T, N, N> ret{};
    for (size_t i = 0; i != N; ++i)
    {
        ret[i, i] = static_cast<T>(1);
    }
    return ret;
}

// template <typename T, size_t N>
// [[nodiscard]] constexpr static_matrix<T, N, N> transpose(static_matrix<T, N, N> const& scr)
// {
//     static_matrix<T, N, N> ret{ scr };
//     for (size_t j = 0; j != N - 1; ++j)
//     {
//         for (size_t i = j + 1; i != N; ++i)
//         {
//             std::swap(ret[j, i], ret[i, j]);
//         }
//     }
//     return ret;
// }

template <typename T, size_t M, size_t N>
[[nodiscard]] constexpr static_matrix<T, N, M> transpose(static_matrix<T, M, N> const& scr)
{
    static_matrix<T, N, M> ret{};
    if constexpr (M == N)
    {
        for (size_t j = 0; j != N - 1; ++j)
        {
            for (size_t i = j + 1; i != N; ++i)
            {
                std::swap(ret[j, i], ret[i, j]);
            }
        }
    }
    else
    {
        for (size_t j = 0; j != M; ++j)
        {
            for (size_t i = 0; i != N; ++i)
            {
                ret[i, j] = scr[j, i];
            }
        }
    }
    return ret;
}

template <std::size_t Out_M, std::size_t Out_N, static_matrix_type Matrix, typename Fn>
    requires((Out_M == 1 || Out_N == 1) &&
             (Out_M == Matrix::Size_y || Out_N == Matrix::Size_x || Out_N == Out_M /* == 1*/) &&
             std::is_invocable_r_v<
                 typename Matrix::value_type,
                 Fn,
                 typename Matrix::value_type,
                 typename Matrix::value_type>)
[[nodiscard]] auto reduce(Matrix const& mat, Fn&& fn) -> static_matrix<typename Matrix::value_type, Out_M, Out_N>
{
    using return_type     = static_matrix<typename Matrix::value_type, Out_M, Out_N>;
    constexpr auto M      = Matrix::Size_y;
    constexpr auto N      = Matrix::Size_x;
    auto           ret    = return_type();
    auto           cbegin = mat.cbegin();
    for (std::size_t j = 0; j != M; ++j)
    {
        const auto out_j = Out_M == 1 ? 0 : j; // j % Out_M
        for (std::size_t i = 0; i != N; ++i)
        {
            const auto out_i  = Out_N == 1 ? 0 : i; // i % Out_N
            ret[out_j, out_i] = fn(std::move(ret[out_j, out_i]), *(cbegin++));
        }
    }
    return ret;
}

template <static_matrix_type Matrix>
[[nodiscard]] constexpr Matrix element_wise_mul(Matrix const& mat1, Matrix const& mat2)
{
    Matrix ret{ mat1 };
    for (size_t j = 0; j != Matrix::M; ++j)
    {
        for (size_t i = 0; i != Matrix::N; ++i)
        {
            ret[j, i] *= mat2[j, i];
        }
    }
    return ret;
}

/// @brief Accumulates dist_op(mat1_ij, mat2_ij) for all ij. If dist_op is the Ln dist_op, then this
/// function returns the Ln dist_op between mat1 and mat2 to the power, i.e. Ln(mat1, mat2)^n
/// @tparam Distance A Binary operation type (R, R) -> R
/// @tparam R A floating point type
/// @tparam Matrix A static matrix type
/// @param mat1
/// @param mat2
/// @param dist_op Binary operation to accumulate over mat1 and mat2
/// @return Distance between mat1 and mat2 defined by the Distance dist_op
template <std::floating_point R, static_matrix_type Matrix, typename Distance>
    requires std::is_invocable_r_v<R, Distance, typename Matrix::value_type, typename Matrix::value_type>
[[nodiscard]] constexpr auto distance(Matrix const& mat1, Matrix const& mat2, Distance&& dist_op) -> R
{
    R distance{};
    for (size_t j = 0; j != Matrix::Size_y; ++j)
    {
        for (size_t i = 0; i != Matrix::Size_x; ++i)
        {
            distance += std::invoke_r<R>(std::forward<Distance>(dist_op), mat1[j, i], mat2[j, i]);
        }
    }
    return distance;
}

//-----------------------------------------------------------------------------
//------------ Miscellany related functionality  -------------------------------------
//-----------------------------------------------------------------------------

/*
returns the element-wise type consistent average of a pack of matrices
beware of overflow issues if matrices have large numbers or calculating the average over a big array
*/
template <static_matrix_type Matrix>
[[nodiscard]] constexpr Matrix matrix_average(std::same_as<Matrix> auto const&... mats)
{
    return (mats + ...) / sizeof...(mats);
}

} /* namespace ga_sm */

#endif // !STATIC_MATRIX
