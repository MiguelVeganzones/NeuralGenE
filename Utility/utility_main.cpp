#include <iostream>

#include "Log.hpp"
#include "Precision_totalizer.hpp"
#include "Random.hpp"
#include "Stopwatch.hpp"
#include "generics.hpp"

auto rng()
{
    auto rn = random::randnormal(0, 10000);
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

int main()
{
    top_n_test();

    MEASURE_FUNCTION_EXECUTION_TIME();
    stopwatch s("name");
    auto      pt  = precision_totalizer{};
    double    sum = 0;

    constexpr std::size_t n = 100'000'000;

    random::init();

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
