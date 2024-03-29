#pragma once

#ifndef STOPWATCH
#define STOPWATCH

#include "Log.hpp"
#include "Precision_totalizer.hpp"
#include "error_handling.hpp"
#include <chrono>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <source_location>
#include <string>

// imeplement stopwatch::timeUnit
// https://github.com/martinus/nanobench/blob/master/src/include/nanobench.h

#define USE_TIMER 1
#if USE_TIMER
#define MEASURE_FUNCTION_EXECUTION_TIME()                                      \
    stopwatch _execution_time_stopwatch_(                                      \
        std::source_location::current().function_name()                        \
    )
#else
#define MEASURE_FUNCTION_EXECUTION_TIME()
#endif

class stopwatch
{
    using clock_type = std::chrono::steady_clock;

public:
    stopwatch(const char* func = "Process") :
        function_name{ func },
        start{ clock_type::now() }
    {
    }

    stopwatch(stopwatch const&)                    = delete;
    stopwatch(stopwatch&&)                         = delete;
    auto operator=(stopwatch const&) -> stopwatch& = delete;
    auto operator=(stopwatch&&) -> stopwatch&      = delete;

    ~stopwatch()
    {
        const auto duration = clock_type::now() - start;
        std::cout << std::fixed << std::setprecision(4);
        std::cout
            << function_name << " took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                   .count()
            << "ms\t ("
            << std::chrono::duration_cast<std::chrono::microseconds>(duration)
                   .count()
            << "us)\t ("
            << std::chrono::duration_cast<std::chrono::seconds>(duration).count(
               )
            << " s)\t ("
            << std::chrono::duration_cast<std::chrono::minutes>(duration).count(
               )
            << " mins)\n"
            << std::defaultfloat;
    }

private:
    const char*                  function_name{};
    const clock_type::time_point start{};
};

/*

time a function :

auto t0 = std::chrono::steady_clock::now();
f();
auto t1 = std::chrono::steady_clock::now();
// std::cout << nanoseconds{t-t0}.count << "ns\n";
// std::cout << std::chrono::duration<double>{t1-t0}.count(); // print in
floating point seconds
// std::cout <<
std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();


*/

class bench
{
public:
    using measurement_units    = std::chrono::nanoseconds;
    using TimeUnits_value_type = int8_t;

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<void, Fn, Args...>
    [[maybe_unused]]
    static measurement_units benchmark(Fn fn, Args&&... args)
    {
        const auto start = std::chrono::steady_clock::now();
        std::invoke(std::forward<Fn>(fn), std::forward<Args>(args)...);
        const auto end = std::chrono::steady_clock::now();

        return end - start;
    }

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<void, Fn, Args...>
    [[maybe_unused]]
    static precision_totalizer multiple_run_bench(
        size_t n,
        Fn     fn,
        Args&&... args
    )
    {
        precision_totalizer totalizer{};

        while (n-- > 0)
        {
            const measurement_units t_ns =
                benchmark(std::forward<Fn>(fn), std::forward<Args>(args)...);
            totalizer.add(t_ns.count());
        }

        print_time_stats(totalizer);

        return totalizer;
    }

private:
    template <typename T, typename S>
    [[nodiscard]]
    static constexpr auto duration_diff(T const& t, S const& s)
        -> std::common_type_t<T, S>
    {
        typedef typename std::common_type_t<T, S> Common;
        return Common(t) - Common(s);
    }

    static void print_time_stats(precision_totalizer const& pt)
    {
        const auto avg =
            measurement_units(static_cast<size_t>(pt.get_average()));
        const auto units     = get_most_meaningful_units(avg);
        const auto units_str = ' ' + s_units_string_identifier[units];
        const auto ratio =
            static_cast<long double>(get_meaningful_ratio_values(units));

        std::cout << "\n-----Benchmark stats------";
        std::cout << "\nRuns:\t\t\t" << pt.get_num_samples();
        std::cout << "\nMin execution time:\t" << pt.get_min() / ratio
                  << units_str;
        std::cout << "\nAverage execution time:\t" << pt.get_average() / ratio
                  << units_str;
        std::cout << "\nStandard deviation:\t" << pt.get_stddev() / ratio
                  << units_str;
        std::cout << "\n-------------------------\n";
    }

private:
    enum TimeUnits : TimeUnits_value_type
    {
        Nanoseconds,
        Microseconds,
        Milliseconds,
        Seconds,
        Minutes,
        Hours
    };

    inline static std::map<TimeUnits, std::string> s_units_string_identifier{
        { Nanoseconds, "ns" }, { Microseconds, "us" }, { Milliseconds, "ms" },
        { Seconds, "s" },      { Minutes, "min" },     { Hours, "h" }
    };

    static TimeUnits get_most_meaningful_units(const measurement_units value)
    {
        if (std::chrono::duration_cast<std::chrono::hours>(value).count() > 0)
            return Hours;
        if (std::chrono::duration_cast<std::chrono::minutes>(value).count() > 0)
            return Minutes;
        if (std::chrono::duration_cast<std::chrono::seconds>(value).count() > 0)
            return Seconds;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(value).count(
            ) > 0)
            return Milliseconds;
        if (std::chrono::duration_cast<std::chrono::microseconds>(value).count(
            ) > 0)
            return Microseconds;
        return Nanoseconds;
    }

    static constexpr std::size_t get_meaningful_ratio_values(TimeUnits units
    ) noexcept
    {
        static_assert(std::is_same_v<
                      measurement_units,
                      std::chrono::nanoseconds>);
        switch (units)
        {
        case Hours:
            return 3'600'000'000'000;
        case Minutes:
            return 60'000'000'000;
        case Seconds:
            return 1'000'000'000;
        case Milliseconds:
            return 1'000'000;
        case Microseconds:
            return 1000;
        case Nanoseconds:
            return 1;
        }
        assert_unreachable();
    }
};

#endif // STOPWATCH
