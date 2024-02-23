#include "Log.hpp"
#include "Precision_totalizer.hpp"
#include "Random.hpp"
#include "Stopwatch.hpp"
#include "generics.hpp"
#include "progressbar.hpp"
#include <chrono>
#include <iostream>
#include <thread>

auto rng()
{
    auto rn = random_::random::s_randnormal(0, 10000);
    return double(rn);
}

auto top_n_test() -> void
{
    auto&& v = std::array{ 0, 555, 5,    2,    888,  322,  53, 3,
                           0, 49,  2143, 213,  2384, 1283, 31, 1383,
                           1, 233, 453,  2323, 429,  -1000 };

    auto&& ret = generics::algorithms::top_n_indeces(v, 10);
    for (auto&& e : ret)
    {
        std::cout << e << ' ';
    }
    std::cout << '\n';
}

void progressbar_main()
{
    progressbar_::progressbar pbar;

    for (int i = 1; i <= 100; i++)
    {
        pbar.print(
            i,
            "\0",
            '\t',
            random_::random::s_randfloat(),
            '\t',
            random_::random::s_randintegral<int>(0, 100),
            '\t'
        );
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

void progressmatrix_main()
{
    random_::random::s_seed();
    progressbar_::progress_matrix pmatrix(5);

    for (int i = 1; i <= 100; i++)
    {
        pmatrix.print(
            random_::random::s_randintegral<int>(0, 4),
            i,
            " ",
            '\t',
            random_::random::s_randfloat(),
            '\t',
            random_::random::s_randintegral<int>(0, 100),
            '\t'
        );
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int totoalizer_main()
{
    top_n_test();

    MEASURE_FUNCTION_EXECUTION_TIME();
    stopwatch s("name");
    auto      pt  = precision_totalizer{};
    double    sum = 0;

    constexpr std::size_t n = 100'000'000;

    random_::random::s_seed();

    for (size_t i = 0; i != n; ++i)
    {
        // auto rn = i;
        auto rn = rng();

        pt.add(rn);
        sum += rn;
    }

    std::cout << std::setprecision(40);
    std::cout << sum << std::endl;
    std::cout << pt.get_integral_value() << std::endl;
    std::cout << pt.get_floating_point_value() << std::endl;

    pt.summary();

    log::add("Testing_0");
    log::add("Sum:\t\t" + std::to_string(sum));
    log::add("Totalizer:\t" + std::to_string(pt.get_value()));
    log::flush_log();

    return EXIT_SUCCESS;
}

int main()
{
    top_n_test();
    progressmatrix_main();
    progressbar_main();
}