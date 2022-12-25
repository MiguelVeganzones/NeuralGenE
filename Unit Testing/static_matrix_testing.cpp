#include "pch.h"

#include "CppUnitTest.h"
#include "Random.h"
#include "static_matrix.hpp"
#include <type_traits>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace NativeUnitTesting
{
TEST_CLASS(unittesting_static_matrix)
{
    static constexpr double epsilon = 1e-5;

    using Mf5  = ga_sm::static_matrix<float, 5, 5>;
    using Md5  = ga_sm::static_matrix<double, 5, 5>;
    using Mi5  = ga_sm::static_matrix<int, 5, 5>;
    using Mf20 = ga_sm::static_matrix<float, 20, 20>;
    using Md20 = ga_sm::static_matrix<double, 20, 20>;
    using Mi20 = ga_sm::static_matrix<int, 20, 20>;

public:
    TEST_METHOD(assert_static_mat_mat_mul)
    {
        constexpr size_t N = 20;

        ga_sm::static_matrix<double, N, N> m1{};

        m1.fill(random::randfloat);

        bool b{ false };

        if (auto [invertible, m1_inv] = inverse(m1); invertible)
        {
            b = (matrix_mul(m1, m1_inv) == ga_sm::identity_matrix<double, N>());
        }

        Assert::AreEqual(b, true);
    }

    TEST_METHOD(assert_static_mat_mat_mul_identity)
    {
        constexpr size_t N = 20;
        using T            = double;

        ga_sm::static_matrix<T, N, N> m{};

        m.fill(random::randfloat);
        const auto I = ga_sm::identity_matrix<T, N>();

        const auto res = matrix_mul(m, I);

        Assert::IsTrue(normalized_L1_distance<double>(m, res) < epsilon);
    }

    TEST_METHOD(assert_static_mat_vec_mul_float)
    {
        constexpr size_t N = 4;

        constexpr ga_sm::static_matrix<float, N, N> m{ 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f,  7.7f,  8.8f,
                                                       9.9f, 3.3f, 4.4f, 2.2f, 3.3f, 14.1f, 15.5f, 0.f };
        constexpr ga_sm::static_matrix<float, N, 1> v{ 3.3f, 4.4f, 2.2f, 2.2f },
            ground_truth{ 30.25f, 83.49f, 61.71f, 107.03f };

        const auto res = ga_sm::matrix_mul(m, v);

        Assert::IsTrue(normalized_L1_distance<double>(res, ground_truth) < epsilon);
    }

    TEST_METHOD(assert_static_mat_vec_mul_int)
    {
        constexpr size_t N = 4;

        constexpr ga_sm::static_matrix<int, N, N> m{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 2, 3, 14, 15, 0 };
        constexpr ga_sm::static_matrix<int, N, 1> v{ 3, 4, 2, 2 }, ground_truth{ 25, 69, 51, 95 };

        const auto res = matrix_mul(m, v);

        Assert::IsTrue(normalized_L1_distance<double>(res, ground_truth) < epsilon);
    }

    TEST_METHOD(assert_static_mat_vec_mul_double)
    {
        constexpr size_t N = 4;

        constexpr ga_sm::static_matrix<double, N, N> m{ 1.1, 2.2, 3.3, 4.4, 5.5, 6.6,  7.7,  8.8,
                                                        9.9, 3.3, 4.4, 2.2, 3.3, 14.1, 15.5, 0. };
        constexpr ga_sm::static_matrix<double, N, 1> v{ 3.3, 4.4, 2.2, 2.2 },
            ground_truth{ 30.25, 83.49, 61.71, 107.03 };

        const auto res = matrix_mul(m, v);

        Assert::IsTrue(normalized_L1_distance<double>(res, ground_truth) < epsilon);
    }

    TEST_METHOD(assert_static_mat_vec_mul_avx)
    {
        constexpr size_t N = 4;

        constexpr ga_sm::static_matrix<float, N, N> m{ 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f,  7.7f,  8.8f,
                                                       9.9f, 3.3f, 4.4f, 2.2f, 3.3f, 14.1f, 15.5f, 0.f };
        constexpr ga_sm::static_matrix<float, N, 1> v{ 3.3f, 4.4f, 2.2f, 2.2f },
            ground_truth{ 30.25f, 83.49f, 61.71f, 107.03f };

        const auto res = matrix_vector_mul_float_avx(m, v);

        Assert::IsTrue(normalized_L1_distance<double>(res, ground_truth) < epsilon);
    }

    TEST_METHOD(assert_static_mat_vec_and_mat_vec_mul_avx)
    {
        constexpr size_t N = 50;

        ga_sm::static_matrix<float, N, N> m{};
        ga_sm::static_matrix<float, N, 1> v{};

        for (int i = 0; i != 10; ++i)
        {
            m.fill(random::randfloat);
            v.fill(random::randfloat);

            const auto res_a = matrix_mul(m, v);
            const auto res_b = matrix_vector_mul_float_avx(m, v);

            Assert::IsTrue(normalized_L1_distance<double>(res_a, res_b) < epsilon);
        }
    }

    TEST_METHOD(assert_x_crossover)
    {
        constexpr int N = 10;
        constexpr int n = N - 1;

        constexpr int K = 20;
        constexpr int k = 10;

        using Mat = ga_sm::static_matrix<int, N, N>;

        // random::init();
        int count = 0;
        for (int i = 0; i != K; ++i)
        {
            Mat mat1{};
            Mat mat2{};

            mat1.fill(random::randint, -10, 10);
            mat2.fill(random::randint, -10, 10);

            const auto [mat12, mat21] = ga_sm::x_crossover(mat1, mat2);

            Assert::IsTrue(mat1 + mat2 == mat12 + mat21);

            if ((mat1(0, 0) == mat12(0, 0) && mat1(n, n) == mat12(n, n) && mat1(0, n) == mat21(0, n) &&
                 mat1(n, 0) == mat21(n, 0) && mat2(0, 0) == mat21(0, 0) && mat2(n, n) == mat21(n, n) &&
                 mat2(0, n) == mat12(0, n) && mat2(n, 0) == mat12(n, 0)) ||
                (mat2(0, 0) == mat12(0, 0) && mat2(n, n) == mat12(n, n) && mat2(0, n) == mat21(0, n) &&
                 mat2(n, 0) == mat21(n, 0) && mat1(0, 0) == mat21(0, 0) && mat1(n, n) == mat21(n, n) &&
                 mat1(0, n) == mat12(0, n) && mat1(n, 0) == mat12(n, 0)))
            {
                ++count;
            }
        }
        Assert::IsTrue(count > k);
    }

    TEST_METHOD(assert_in_place_x_crossover)
    {
        constexpr int N = 10;
        constexpr int n = N - 1;

        constexpr int K = 20;
        constexpr int k = 10;

        using Mat = ga_sm::static_matrix<int, N, N>;

        // random::init();
        int count = 0;
        for (int i = 0; i != K; ++i)
        {
            Mat mat12{};
            Mat mat21{};

            mat12.fill(random::randint, -10, 10);
            mat21.fill(random::randint, -10, 10);

            const Mat mat1{ mat12 };
            const Mat mat2{ mat21 };

            ga_sm::in_place_x_crossover(mat12, mat21);

            Assert::IsTrue(mat1 + mat2 == mat12 + mat21);

            if ((mat1(0, 0) == mat12(0, 0) && mat1(n, n) == mat12(n, n) && mat1(0, n) == mat21(0, n) &&
                 mat1(n, 0) == mat21(n, 0) && mat2(0, 0) == mat21(0, 0) && mat2(n, n) == mat21(n, n) &&
                 mat2(0, n) == mat12(0, n) && mat2(n, 0) == mat12(n, 0)) ||
                (mat2(0, 0) == mat12(0, 0) && mat2(n, n) == mat12(n, n) && mat2(0, n) == mat21(0, n) &&
                 mat2(n, 0) == mat21(n, 0) && mat1(0, 0) == mat21(0, 0) && mat1(n, n) == mat21(n, n) &&
                 mat1(0, n) == mat12(0, n) && mat1(n, 0) == mat12(n, 0)))
            {
                ++count;
            }
        }
        Assert::IsTrue(count > k);
    }

    TEST_METHOD(assert_to_target_x_crossover)
    {
        constexpr int N = 10;
        constexpr int n = N - 1;

        constexpr int K = 20;
        constexpr int k = 10;

        using Mat = ga_sm::static_matrix<int, N, N>;

        // random::init();
        int count = 0;
        for (int i = 0; i != K; ++i)
        {
            Mat mat1{};
            Mat mat2{};
            Mat mat12{};
            Mat mat21{};

            mat1.fill(random::randint, -10, 10);
            mat2.fill(random::randint, -10, 10);
            mat12.fill(random::randint, -10, 10);
            mat21.fill(random::randint, -10, 10);

            ga_sm::to_target_x_crossover(mat1, mat2, mat12, mat21);

            Assert::IsTrue(mat1 + mat2 == mat12 + mat21);

            if ((mat1(0, 0) == mat12(0, 0) && mat1(n, n) == mat12(n, n) && mat1(0, n) == mat21(0, n) &&
                 mat1(n, 0) == mat21(n, 0) && mat2(0, 0) == mat21(0, 0) && mat2(n, n) == mat21(n, n) &&
                 mat2(0, n) == mat12(0, n) && mat2(n, 0) == mat12(n, 0)) ||
                (mat2(0, 0) == mat12(0, 0) && mat2(n, n) == mat12(n, n) && mat2(0, n) == mat21(0, n) &&
                 mat2(n, 0) == mat21(n, 0) && mat1(0, 0) == mat21(0, 0) && mat1(n, n) == mat21(n, n) &&
                 mat1(0, n) == mat12(0, n) && mat1(n, 0) == mat12(n, 0)))
            {
                ++count;
            }
        }
        Assert::IsTrue(count > k);
    }

    TEST_METHOD(assert_x_crossover_rectangular)
    {
        constexpr int N = 10;
        constexpr int n = N - 1;

        constexpr int M = 15;
        constexpr int m = M - 1;

        constexpr int K = 20;
        constexpr int k = 12;

        using Mat = ga_sm::static_matrix<int, M, N>;

        // random::init();
        int count = 0;
        for (int i = 0; i != K; ++i)
        {
            Mat mat1{};
            Mat mat2{};

            mat1.fill(random::randint, -10, 10);
            mat2.fill(random::randint, -10, 10);

            const auto [mat12, mat21] = ga_sm::x_crossover(mat1, mat2);

            Assert::IsTrue(mat1 + mat2 == mat12 + mat21);

            if ((mat1(0, 0) == mat12(0, 0) && mat1(m, n) == mat12(m, n) && mat1(0, n) == mat21(0, n) &&
                 mat1(m, 0) == mat21(m, 0) && mat2(0, 0) == mat21(0, 0) && mat2(m, n) == mat21(m, n) &&
                 mat2(0, n) == mat12(0, n) && mat2(m, 0) == mat12(m, 0)) ||
                (mat2(0, 0) == mat12(0, 0) && mat2(m, n) == mat12(m, n) && mat2(0, n) == mat21(0, n) &&
                 mat2(m, 0) == mat21(m, 0) && mat1(0, 0) == mat21(0, 0) && mat1(m, n) == mat21(m, n) &&
                 mat1(0, n) == mat12(0, n) && mat1(m, 0) == mat12(m, 0)))
            {
                ++count;
            }
        }
        Assert::IsTrue(count > k);
    }

    TEST_METHOD(assert_in_place_x_crossover_rectangular)
    {
        constexpr int N = 10;
        constexpr int n = N - 1;

        constexpr int M = 15;
        constexpr int m = M - 1;

        constexpr int K = 20;
        constexpr int k = 12;

        using Mat = ga_sm::static_matrix<int, M, N>;

        // random::init();
        int count = 0;
        for (int i = 0; i != K; ++i)
        {
            Mat mat12{};
            Mat mat21{};

            mat12.fill(random::randint, -10, 10);
            mat21.fill(random::randint, -10, 10);

            const Mat mat1{ mat12 };
            const Mat mat2{ mat21 };

            ga_sm::in_place_x_crossover(mat12, mat21);

            Assert::IsTrue(mat1 + mat2 == mat12 + mat21);

            if ((mat1(0, 0) == mat12(0, 0) && mat1(m, n) == mat12(m, n) && mat1(0, n) == mat21(0, n) &&
                 mat1(m, 0) == mat21(m, 0) && mat2(0, 0) == mat21(0, 0) && mat2(m, n) == mat21(m, n) &&
                 mat2(0, n) == mat12(0, n) && mat2(m, 0) == mat12(m, 0)) ||
                (mat2(0, 0) == mat12(0, 0) && mat2(m, n) == mat12(m, n) && mat2(0, n) == mat21(0, n) &&
                 mat2(m, 0) == mat21(m, 0) && mat1(0, 0) == mat21(0, 0) && mat1(m, n) == mat21(m, n) &&
                 mat1(0, n) == mat12(0, n) && mat1(m, 0) == mat12(m, 0)))
            {
                ++count;
            }
        }
        Assert::IsTrue(count > k);
    }

    TEST_METHOD(assert_to_target_x_crossover_rectangular)
    {
        constexpr int N = 10;
        constexpr int n = N - 1;

        constexpr int M = 15;
        constexpr int m = M - 1;

        constexpr int K = 20;
        constexpr int k = 12;

        using Mat = ga_sm::static_matrix<int, M, N>;

        // random::init();
        int count = 0;
        for (int i = 0; i != K; ++i)
        {
            Mat mat1{};
            Mat mat2{};
            Mat mat12{};
            Mat mat21{};

            mat1.fill(random::randint, -10, 10);
            mat2.fill(random::randint, -10, 10);
            mat12.fill(random::randint, -10, 10);
            mat21.fill(random::randint, -10, 10);

            ga_sm::to_target_x_crossover(mat1, mat2, mat12, mat21);

            Assert::IsTrue(mat1 + mat2 == mat12 + mat21);

            if ((mat1(0, 0) == mat12(0, 0) && mat1(m, n) == mat12(m, n) && mat1(0, n) == mat21(0, n) &&
                 mat1(m, 0) == mat21(m, 0) && mat2(0, 0) == mat21(0, 0) && mat2(m, n) == mat21(m, n) &&
                 mat2(0, n) == mat12(0, n) && mat2(m, 0) == mat12(m, 0)) ||
                (mat2(0, 0) == mat12(0, 0) && mat2(m, n) == mat12(m, n) && mat2(0, n) == mat21(0, n) &&
                 mat2(m, 0) == mat21(m, 0) && mat1(0, 0) == mat21(0, 0) && mat1(m, n) == mat21(m, n) &&
                 mat1(0, n) == mat12(0, n) && mat1(m, 0) == mat12(m, 0)))
            {
                ++count;
            }
        }
        Assert::IsTrue(count > k);
    }

    TEST_METHOD(assert_is_trivially_copiable_float)
    {
        Assert::IsTrue(std::is_trivially_copyable_v<Mf5>);
    }
    TEST_METHOD(assert_is_trivial_float)
    {
        Assert::IsTrue(std::is_trivial_v<Mf5>);
    }
    TEST_METHOD(assert_is_standard_layout_float)
    {
        Assert::IsTrue(std::is_standard_layout_v<Mf5>);
    }

    TEST_METHOD(assert_is_trivially_copiable_int)
    {
        Assert::IsTrue(std::is_trivially_copyable_v<Mi5>);
    }
    TEST_METHOD(assert_is_trivial_int)
    {
        Assert::IsTrue(std::is_trivial_v<Mi5>);
    }
    TEST_METHOD(assert_is_standard_layout_int)
    {
        Assert::IsTrue(std::is_standard_layout_v<Mi5>);
    }

    TEST_METHOD(assert_is_trivially_copiable_double)
    {
        Assert::IsTrue(std::is_trivially_copyable_v<Md5>);
    }
    TEST_METHOD(assert_is_trivial_double)
    {
        Assert::IsTrue(std::is_trivial_v<Md5>);
    }
    TEST_METHOD(assert_is_standard_layout_double)
    {
        Assert::IsTrue(std::is_standard_layout_v<Md5>);
    }
};
} // namespace NativeUnitTesting
