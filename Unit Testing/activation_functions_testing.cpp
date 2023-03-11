#include "pch.h"

#include "CppUnitTest.h"
#include "activation_functions.hpp"
#include "static_matrix.hpp"

#include <type_traits>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace NativeUnitTesting
{
TEST_CLASS(unittesting_activation_functions)
{
    using Mf5  = ga_sm::static_matrix<float, 5, 5>;
    using Md5  = ga_sm::static_matrix<double, 5, 5>;
    using Mi5  = ga_sm::static_matrix<int, 5, 5>;
    using Mf20 = ga_sm::static_matrix<float, 20, 20>;
    using Md20 = ga_sm::static_matrix<double, 20, 20>;
    using Mi20 = ga_sm::static_matrix<int, 20, 20>;

    using AF_ReLU = matrix_activation_functions::activation_function<Mf5, matrix_activation_functions::Identifiers::ReLU>;

public:
    TEST_METHOD(assert_is_trivially_copiable)
    {
        Assert::IsTrue(std::is_trivially_copyable_v<AF_ReLU>);
    }
    TEST_METHOD(assert_is_trivial)
    {
        Assert::IsTrue(std::is_trivial_v<AF_ReLU>);
    }
    TEST_METHOD(assert_is_standard_layout)
    {
        Assert::IsTrue(std::is_standard_layout_v<AF_ReLU>);
    }
    TEST_METHOD(assert_is_trivially_default_constructible)
    {
        Assert::IsTrue(std::is_trivially_default_constructible_v<AF_ReLU>);
    }

    TEST_METHOD(assert_is_trivially_constructible)
    {
        Assert::IsTrue(std::is_trivially_constructible_v<AF_ReLU>);
    }
};
} // namespace NativeUnitTesting
