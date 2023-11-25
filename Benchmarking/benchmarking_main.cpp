#include <benchmark/benchmark.h>
#include <cmath>

static void BM_StringCreation(benchmark::State& state)
{
    for (auto _ : state)
    {
        std::string empty_string;
    }
}

// Register the function as benchmark
BENCHMARK(BM_StringCreation);

// Define another benchmark
static void BM_StringCopy(benchmark::State& state)
{
    std::string x = "hello";
    for (auto _ : state)
    {
        std::string copy(x);
    }
}

BENCHMARK(BM_StringCopy);

static void std_Pow(benchmark::State& state)
{
    for (auto _ : state)
    {
        for (float i = 0; i < 1000; ++i)
        {
            [[maybe_unused]] auto a = std::pow(i, 2);
        }
    }
}

static void hand_mul(benchmark::State& state)
{
    for (auto _ : state)
    {
        for (float i = 0; i < 1000; ++i)
        {
            [[maybe_unused]] auto a = i * i;
        }
    }
}

BENCHMARK(std_Pow);
BENCHMARK(hand_mul);


BENCHMARK_MAIN();