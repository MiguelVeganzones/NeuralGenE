#pragma once

#ifndef RANDOM_NUMBER_GENERATOR
#define RANDOM_NUMBER_GENERATOR

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <random>
// #include <mutex>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <limits>

#ifdef max
#undef max
#endif

class random
{
public:
    inline static void init()
    {
        s_Random_engine.seed(
            static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    }

    inline static float randfloat()
    {
        const auto gen = static_cast<float>(s_Distribution(random::s_Random_engine));
        return gen / int32_max_float;
    }

    inline static int randint(const int min, const int max)
    {
        std::uniform_int_distribution<int> u(min, max);
        return u(random::s_Random_engine);
    }

    inline static size_t randsize_t(const size_t min, const size_t max)
    {
        assert(min <= max);  
        std::uniform_int_distribution<size_t> u(min, max);
        return u(random::s_Random_engine);
    }

    inline static float randnormal(const float avg = 0.f, const float stddev = 1.f)
    {
        std::normal_distribution<float> n(avg, stddev);
        return n(random::s_Random_engine);
    }

    // static int mt_randint(int Min, int Max);

    // static float mt_randfloat();

    // static float mt_randnormal(const float avg = 0.f, const float stddev = 1.f);

private:
    inline static std::mt19937                                             s_Random_engine;
    inline static std::uniform_int_distribution<std::mt19937::result_type> s_Distribution;
    inline static constexpr float int32_max_float = static_cast<float>(std::numeric_limits<uint32_t>::max());
};

// std::mutex mu_randint;
// std::mutex mu_randfloat;
// std::mutex mu_randnormal;

// void random::init()
//{
//   s_random_engine.seed((unsigned
//   int)std::chrono::high_resolution_clock::now().time_since_epoch().count());
// }
//
// float random::randfloat()
//{
//   return (float)s_distribution(random::s_random_engine) /
//   (float)std::numeric_limits<uint32_t>::max();
// }

// float random::mt_randfloat()
//{
//   mu_randfloat.lock();
//   const float f = (float)s_distribution(random::s_random_engine) /
//   (float)std::numeric_limits<uint32_t>::max(); mu_randfloat.unlock(); return f;
// }

// int random::randint(int Min = 0, int Max = 10)
//{
//   std::uniform_int_distribution<int> U(Min, Max);
//   return U(random::s_random_engine);
// }

// int random::mt_randint(int Min = 0, int Max = 10) //multi threaded randint
//{
//   std::uniform_int_distribution<int> U(Min, Max);
//   mu_randint.lock();
//   const int n = U(random::s_random_engine);
//   mu_randint.unlock();
//   return n;
// }

// float random::randnormal(const float avg, const float stddev) {
//   std::normal_distribution<float> N(avg, stddev);
//   return N(random::s_random_engine);
// }

// float random::mt_randnormal(const float avg, const float stddev) {
//   std::normal_distribution<float> N(avg, stddev);
//   mu_randnormal.lock();
//   const float n = N(random::s_random_engine);
//   mu_randnormal.unlock();
//   return n;
// }

#endif // RANDOM_NUMBER_GENERATOR
