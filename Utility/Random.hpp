#pragma once

#ifndef RANDOM_NUMBER_GENERATOR
#define RANDOM_NUMBER_GENERATOR

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>

#ifdef max
#undef max
#endif

namespace random_
{
class random
{
private:
    [[nodiscard]]
    static auto static_instance() noexcept -> random&
    {
        static random s_Random_engine(0);
        return s_Random_engine;
    }

public:
    static auto s_seed(unsigned int seed_ = std::random_device{}()) noexcept
        -> void
    {
        static_instance().seed(seed_);
    }

    [[nodiscard]]
    static auto s_randfloat() noexcept -> float
    {
        return static_instance().randfloat();
    }

    /// @brief Generates a uniform integral number in the range [min, max]
    /// @tparam T Integral type
    /// @param min Inclusive lower bound
    /// @param max Inclusive upper bound
    /// @return integral number of type T uniformly distributed in the raneg
    /// [min, max]
    template <std::integral T>
    [[nodiscard]]
    static auto s_randintegral(T min, T max) noexcept -> T
    {
        return static_instance().randintegral(min, max);
    }

    [[nodiscard]]
    inline static auto s_randnormal(float avg, float stddev) noexcept -> float
    {
        return static_instance().randnormal(avg, stddev);
    }

    [[nodiscard]]
    static auto s_standard_randnormal() noexcept -> float
    {
        return static_instance().randnormal();
    }

public:
    random(unsigned int seed_ = std::random_device{}()) noexcept
    {
        seed(seed_);
    }

    random(random const&) noexcept            = default;
    random(random&&) noexcept                 = default;
    random& operator=(random const&) noexcept = default;
    random& operator=(random&&) noexcept      = default;
    ~random() noexcept                        = default;

    [[nodiscard]]
    auto randfloat() noexcept -> float
    {
        return m_Uniform_real(m_Random_engine);
    }

    /// @brief Generates a uniform integral number in the range [min, max]
    /// @tparam T Integral type
    /// @param min Inclusive lower bound
    /// @param max Inclusive upper bound
    /// @return integral number of type T uniformly distributed in the raneg
    /// [min, max]
    template <std::integral T>
    [[nodiscard]]
    auto randintegral(T min, T max) noexcept -> T
    {
        assert(min <= max);
        std::uniform_int_distribution<int> u(min, max);
        return u(random::m_Random_engine);
    }

    [[nodiscard]]
    auto randnormal(float avg, float stddev) noexcept -> float
    {
        std::normal_distribution<float> n(avg, stddev);
        return n(m_Random_engine);
    }

    [[nodiscard]]
    auto randnormal() noexcept -> float
    {
        return m_Default_normal(m_Random_engine);
    }

private:
    inline auto seed(unsigned int seed) noexcept -> void
    {
        m_Random_engine.seed(seed);
    }

private:
    std::mt19937                          m_Random_engine;
    std::uniform_real_distribution<float> m_Uniform_real =
        std::uniform_real_distribution<float>(0.f, 1.f);
    std::normal_distribution<float> m_Default_normal =
        std::normal_distribution<float>(0.f, 1.f);
};
} // namespace random_
#endif // RANDOM_NUMBER_GENERATOR
