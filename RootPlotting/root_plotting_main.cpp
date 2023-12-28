#include "TApplication.h"
#include "root_plotting_utility.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <thread>

void test0()
{
    static constexpr std::size_t N = 1000;

    float x[N];
    std::iota(std::begin(x), std::end(x), 0.f);
    float y1[N];
    float y2[N];
    for (int i = 0; i != N; ++i)
    {
        const auto _x = x[i] / 50.f;
        y1[i]         = std::sin(_x) * std::cos(std::sqrt(_x));
        y2[i]         = std::sin(_x) * std::cos(_x);
    }

    root_plotting::plot(N, std::begin(x), std::begin(y1), std::begin(y2));
}

void test1()
{
    static constexpr std::size_t N = 1000;

    float x[N];
    std::iota(std::begin(x), std::end(x), 0.f);
    float y1[N];
    float y2[N];
    for (int i = 0; i != N; ++i)
    {
        const auto _x = x[i] / 50.f;
        y1[i]         = std::sin(_x) * std::cos(std::sqrt(_x));
        y2[i]         = std::sin(_x) * std::cos(_x);
    }
    root_plotting::progress_plot graph(N);
    graph.plot(std::begin(x), std::begin(y1), std::begin(y2));
    for (int i = 0; i != N; i += 10)
    {
        for (int j = 0; j != N; ++j)
        {
            const auto _x = (x[j] + (float)i) / 50.f;
            y2[j]         = std::sin(_x) * std::cos(_x);
            // graph.update();
        }
        graph.update_plot2(std::begin(y2));
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(100ms);
    }
}

int main()
{
    test1();
}