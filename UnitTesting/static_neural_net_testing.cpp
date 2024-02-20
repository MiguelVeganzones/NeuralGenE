#include "pch.h"

#include <filesystem>
#include <functional>
#include <string>
#include <type_traits>

#include "CppUnitTest.h"
#include "Random.hpp"
#include "activation_functions.hpp"
#include "static_matrix.hpp"
#include "static_neural_net.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace NativeUnitTesting
{
TEST_CLASS(unittesting_static_neural_net)
{
public:
    inline static const std::filesystem::path output_file =
        "../../Unit Testing/buffer.txt";

    inline static constexpr double epsilon = 1e-5;

    inline static const matrix_activation_functions::
        ActivationFunctionIdentifiers::PReLU =
            matrix_activation_functions::ActivationFunctionIdentifiers::PReLU;
    inline static const matrix_activation_functions::
        ActivationFunctionIdentifiers::UnsignedSigmoid =
            matrix_activation_functions::ActivationFunctionIdentifiers::
                UnsignedSigmoid;

    inline static constexpr ga_snn::Layer_Signature a3{ 9, PReLU };
    inline static constexpr ga_snn::Layer_Signature a4{ 16, UnsignedSigmoid };

    using T  = float;
    using N  = ga_snn::static_neural_net<T, 1, a3, a3, a3, a4>;
    using L  = ga_snn::layer<T, 1, ga_snn::Layer_Structure{ 3, 3, PReLU }>;
    using LU = ga_snn::layer_unroll<T, 3, 1, a3>;

    TEST_METHOD(assert_init_from_ptr)
    {
        random::init();

        auto uptr1 =
            ga_snn::static_neural_net_factory<N>(random::randnormal, 0, 1);
        auto uptr2 = std::make_unique<N>();

        uptr2->init_from_ptr(uptr1.get());

        Assert::IsTrue(
            ga_snn::L1_net_distance<double>(*uptr1.get(), *uptr2.get()) == 0.0
        );
    }

    TEST_METHOD(assert_store_and_load)
    {
        auto uptr1 =
            ga_snn::static_neural_net_factory<N>(random::randnormal, 0, 1);
        uptr1->store(output_file);

        auto uptr2 = std::make_unique<N>();
        uptr2->load(output_file);

        Assert::IsTrue(
            ga_snn::L1_net_distance<double>(*uptr1.get(), *uptr2.get()) <
            epsilon
        );
    }

    TEST_METHOD(assert_population_variability_same_net)
    {
        const auto uptr1 =
            ga_snn::static_neural_net_factory<N>(random::randnormal, 0, 1);
        const auto uptr2 = std::make_unique<N>(*uptr1);
        const auto uptr3 = std::make_unique<N>(*uptr1);
        const auto uptr4 = std::make_unique<N>(*uptr1);

        const auto ret =
            ga_snn::population_variability<double, 4, N>({ std::cref(*uptr1),
                                                           std::cref(*uptr2),
                                                           std::cref(*uptr3),
                                                           std::cref(*uptr4) });

        for (int j = 0; j != 4; ++j)
        {
            for (int i = 0; i != 4; ++i)
            {
                Assert::IsTrue(ret(j, i) < epsilon);
            }
        }
    }

    TEST_METHOD(assert_population_variability_different_net)
    {
        auto uptr1 =
            ga_snn::static_neural_net_factory<N>(random::randnormal, 0, 1);
        auto uptr2 =
            ga_snn::static_neural_net_factory<N>(random::randnormal, 0, 1);
        auto uptr3 =
            ga_snn::static_neural_net_factory<N>(random::randnormal, 0, 1);
        auto uptr4 =
            ga_snn::static_neural_net_factory<N>(random::randnormal, 0, 1);

        const auto ret =
            ga_snn::population_variability<double, 4, N>({ std::cref(*uptr1),
                                                           std::cref(*uptr2),
                                                           std::cref(*uptr3),
                                                           std::cref(*uptr4) });

        for (int j = 0; j != 4; ++j)
        {
            for (int i = 0; i != 4; ++i)
            {
                Assert::IsTrue(ret(j, i) >= (j == i ? 0. : epsilon));
            }
        }
    }

    TEST_METHOD(assert_staticneural_net_size)
    {
        Assert::IsTrue(sizeof(N) == N::subnet_size());
    }

    TEST_METHOD(assert_is_trivially_copiable_nnet)
    {
        Assert::IsTrue(std::is_trivially_copyable_v<N>);
    }

    TEST_METHOD(assert_is_trivially_default_constructible_nnet)
    {
        Assert::IsTrue(std::is_trivially_default_constructible_v<N>);
    }

    TEST_METHOD(assert_is_trivially_constructible_nnet)
    {
        Assert::IsTrue(std::is_trivially_constructible_v<N>);
    }

    TEST_METHOD(assert_is_trivially_default_destructible_nnet)
    {
        Assert::IsTrue(std::is_trivially_destructible_v<N>);
    }

    TEST_METHOD(assert_is_trivially_default_copy_constructible_nnet)
    {
        Assert::IsTrue(std::is_trivially_copy_constructible_v<N>);
    }

    TEST_METHOD(assert_is_trivially_default_copy_assignable_nnet)
    {
        Assert::IsTrue(std::is_trivially_copy_assignable_v<N>);
    }

    TEST_METHOD(assert_is_trivially_default_move_constructible_nnet)
    {
        Assert::IsTrue(std::is_trivially_move_constructible_v<N>);
    }

    TEST_METHOD(assert_is_trivially_default_move_assignable_nnet)
    {
        Assert::IsTrue(std::is_trivially_move_assignable_v<N>);
    }
    TEST_METHOD(assert_is_trivial_nnet)
    {
        Assert::IsTrue(std::is_trivial_v<N>);
    }
    TEST_METHOD(assert_is_standard_layout_nnet)
    {
        Assert::IsTrue(std::is_standard_layout_v<N>);
    }

    TEST_METHOD(assert_is_standard_layout_layer)
    {
        Assert::IsTrue(std::is_standard_layout_v<L>);
    }

    TEST_METHOD(assert_is_trivial_layer)
    {
        Assert::IsTrue(std::is_trivial_v<L>);
    }

    TEST_METHOD(assert_is_trivially_constructible_layer)
    {
        Assert::IsTrue(std::is_trivially_constructible_v<L>);
    }

    TEST_METHOD(assert_is_trivially_default_constructible_layer)
    {
        Assert::IsTrue(std::is_trivially_default_constructible_v<L>);
    }

    TEST_METHOD(assert_is_trivially_copy_constructible_layer)
    {
        Assert::IsTrue(std::is_trivially_copy_constructible_v<L>);
    }

    TEST_METHOD(assert_is_trivially_destructible_layer)
    {
        Assert::IsTrue(std::is_trivially_destructible_v<L>);
    }

    TEST_METHOD(assert_is_trivial_layer_unroll)
    {
        Assert::IsTrue(std::is_trivial_v<LU>);
    }

    TEST_METHOD(assert_is_standard_layout_layer_unroll)
    {
        Assert::IsTrue(std::is_standard_layout_v<LU>);
    }
};
} // namespace NativeUnitTesting
