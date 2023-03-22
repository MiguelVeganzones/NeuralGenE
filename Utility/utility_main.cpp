#include <iostream>

#include "Log.h"
#include "Precision_totalizer.h"
#include "Random.h"

auto rng()
{
    auto rn = random::randnormal(0, 10000);
    return double(rn);
}

int main()
{
    auto   pt  = precision_totalizer{};
    double sum = 0;

    constexpr size_t n = 10'000'000;

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
