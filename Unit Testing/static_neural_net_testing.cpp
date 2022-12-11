#include "pch.h"

#include "CppUnitTest.h"
#include "Random.h"
#include "activation_functions.hpp"
#include "static_matrix.hpp"
#include "static_neural_net.hpp"

#include <type_traits>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace NativeUnitTesting
{
TEST_CLASS(unittesting_static_neural_net)
{
public:
    static const matrix_activation_functions::Identifiers AF = matrix_activation_functions::Identifiers::Sigmoid;
    static constexpr ga_snn::Layer_Signature              a3{ 9, AF };
    using T  = float;
    using N  = ga_snn::static_neural_net<T, 1, a3, a3, a3>;
    using L  = ga_snn::layer<T, 1, ga_snn::Layer_Structure{ 3, 3, AF }>;
    using LU = ga_snn::layer_unroll<T, 3, 1, a3>;

    TEST_METHOD(assert_init_from_ptr)
    {
        random::init();

        auto uptr1 = ga_snn::static_neural_net_factory<N>(random::randnormal, 0, 1);
        auto uptr2 = std::make_unique<N>();

        uptr2->init_from_ptr(uptr1.get());

        Assert::IsTrue(ga_snn::L11_net_distance(uptr1.get(), uptr2.get()) == 0.0);
    }

    TEST_METHOD(assert_staticneural_net_size)
    {
        Assert::IsTrue(sizeof(N) == N::parameter_count() * sizeof(T));
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
