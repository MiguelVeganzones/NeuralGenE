#pragma once

#include <array>
#include <vector>
#include <filesystem>
#include <concepts>
#include <future>
#include <thread>

#include "Random.h"
#include "Stopwatch.h"
#include "activation_functions.hpp"
#include "neural_model.hpp"
#include "score_functions.hpp"
#include "static_matrix.hpp"
#include "static_neural_net.hpp"


void score_functions_test()
{
    constexpr size_t N = 10;

    using namespace ga_snn;
    using namespace ga_sm;

    constexpr auto fn = score_functions::score_functions<double, std::uint16_t>::
        choose_function<score_functions::Identifiers::Weighted_normalized_score, 3, 1, 0>();

    constexpr auto AF = matrix_activation_functions::Identifiers::ReLU;
    auto           SM =
        score_function_objects::score_function_object<decltype(fn), std::uint16_t, std::uint16_t, std::uint16_t>(fn);


    using SM_t = decltype(SM);

    constexpr Layer_Signature a1{ 1, AF };
    constexpr Layer_Signature a9{ 9, AF };
    constexpr Layer_Signature a25{ 25, AF };

    using NET = static_neural_net<float, 1, a1, a9, a25, a25, a25, a25, a9, a1>;

    auto brain_a = ga_neural_model::brain<NET, SM_t>(random::randnormal, 0, 1);

    std::array<ga_neural_model::brain<NET, SM_t>, N> gen_a{}, gen_b{};

    for (size_t i = 0; i != N; ++i)
    {
        gen_a[i] = ga_neural_model::brain<NET, SM_t>(SM, random::randnormal, 0, 1);
        gen_b[i] = ga_neural_model::brain<NET, SM_t>(SM, random::randnormal, 0, 1);
    }

    std::cout << gen_a[0].get_score_function_obj()->operator()() << std::endl;
    gen_a[0].get_score_function_obj()->lost();
    gen_a[0].get_score_function_obj()->tied();
    gen_a[0].get_score_function_obj()->won();
    std::cout << gen_a[0].get_score_function_obj()->operator()() << std::endl;
    std::cout << gen_a[0].get_score() << std::endl;
}

void init_plotting(const size_t len)
{
    const std::filesystem::path py_path   = "../Python Plotting/Python_Plotting.py";
    const std::filesystem::path data_path = "../__Plotting_files/plotting_numbers.txt";

    std::stringstream s;

    s << "python " << py_path << ' ' << data_path << ' ' << len;

    system(s.str().c_str());
}

template<typename Iterable>
void output_to_file(const size_t gen, const Iterable& iter)
{
    const std::filesystem::path data_path = "../__Plotting_files/plotting_numbers.txt";
    std::ofstream               f(data_path, std::ofstream::out | std::ofstream::trunc);

    if (!f.is_open())
        throw;
    f << gen << '\n';
    f << std::setprecision(2);
    for (const float e : iter)
        f << e << ' ';
    f.close();
}

void training_test()
{
    random::init();

    constexpr size_t N = 20;
    constexpr size_t M = 2000;

    //std::thread(init_plotting, M).detach();

    using namespace ga_snn;
    using namespace ga_sm;

    constexpr auto fn = score_functions::score_functions<double, std::uint16_t>::
        choose_function<score_functions::Identifiers::Weighted_normalized_score, 3, 1, 0>();

    constexpr auto AF_relu = matrix_activation_functions::Identifiers::ReLU;
    constexpr auto AF_tanh = matrix_activation_functions::Identifiers::Tanh;
    auto           SM =
        score_function_objects::score_function_object<decltype(fn), std::uint16_t, std::uint16_t, std::uint16_t>(fn);


    using SM_t = decltype(SM);

    constexpr Layer_Signature a1{ 1, AF_relu };
    constexpr Layer_Signature a1_tanh{ 1, AF_tanh };
    constexpr Layer_Signature a9{ 9, AF_relu };
    constexpr Layer_Signature a25{ 25, AF_relu };
    constexpr Layer_Signature a42{ 42, AF_relu };
    constexpr Layer_Signature a16{ 16, AF_relu };

    using NET2 = static_neural_net<float, 1, a1, a42, a16, a1_tanh>;
    using NET = static_neural_net<float, 1, a1, a25, a25, a9, a9, a25, a25, a9, a1_tanh>;

    std::cout << NET::parameter_count() << std::endl;
    //std::cout << NET2::parameter_count() << std::endl;

    std::array<std::array<ga_neural_model::brain<NET, SM_t>, N>, 2> gen{};

    for (size_t i = 0; i != N; ++i)
    {
        gen[0][i] = ga_neural_model::brain<NET, SM_t>(SM, [] { return random::randfloat() * 0.001f; });
        gen[1][i] = ga_neural_model::brain<NET, SM_t>(SM, [] { return 0; });
    }

    std::vector<float> in(M), out(M), pred(M), best_pred(M);
    std::iota(in.begin(), in.end(), 0.f);
    std::ranges::transform(in, out.begin(), [](float i) { return std::sin(i / 100); });

    for (size_t iter = 0; iter != 100000; ++iter)
    {
        std::array<float, N> error{ };
        for (size_t j = 0; j != N; ++j)
        {
            for (size_t i = 0; i != M; ++i)
            {
                pred[i] = gen[iter % 2][j].weigh(in[i]);
                error[j] += std::abs(pred[i] - out[i]);
            }
            //if (j == 0 || (j > 0 && error[j] < error[j - 1]))
            //{
            //    best_pred = pred;
            //}
        }

        // for (const auto e :error)
        //{
        //     std::cout << e << ' ';
        // }
        // std::cout << '\n';

        auto temp = std::array(error);
        std::ranges::sort(temp);

        const auto idx0 = std::ranges::find(error, temp[0]) - error.begin();
        const auto idx1 = std::ranges::find(error, temp[1]) - error.begin();
        const auto idx2 = std::ranges::find(error, temp[2]) - error.begin();
        const auto idx3 = std::ranges::find(error, temp[3]) - error.begin();
        const auto idx4 = std::ranges::find(error, temp[4]) - error.begin();
        const auto idx5 = std::ranges::find(error, temp[5]) - error.begin();

        to_target_brain_x_crossover(
            gen[iter % 2][idx0], gen[iter % 2][idx1], gen[(iter + 1) % 2][0], gen[(iter + 1) % 2][1]);
        to_target_brain_x_crossover(
            gen[iter % 2][idx0], gen[iter % 2][idx2], gen[(iter + 1) % 2][2], gen[(iter + 1) % 2][3]);
        to_target_brain_x_crossover(
            gen[iter % 2][idx0], gen[iter % 2][idx3], gen[(iter + 1) % 2][4], gen[(iter + 1) % 2][5]);
        to_target_brain_x_crossover(
            gen[iter % 2][idx0], gen[iter % 2][idx4], gen[(iter + 1) % 2][6], gen[(iter + 1) % 2][7]);
        to_target_brain_x_crossover(
            gen[iter % 2][idx0], gen[iter % 2][idx5], gen[(iter + 1) % 2][8], gen[(iter + 1) % 2][9]);
        to_target_brain_x_crossover(
            gen[iter % 2][idx1], gen[iter % 2][idx2], gen[(iter + 1) % 2][10], gen[(iter + 1) % 2][11]);
        to_target_brain_x_crossover(
            gen[iter % 2][idx1], gen[iter % 2][idx3], gen[(iter + 1) % 2][12], gen[(iter + 1) % 2][13]);
        to_target_brain_x_crossover(
            gen[iter % 2][idx1], gen[iter % 2][idx4], gen[(iter + 1) % 2][14], gen[(iter + 1) % 2][15]);
        to_target_brain_x_crossover(
            gen[iter % 2][idx2], gen[iter % 2][idx3], gen[(iter + 1) % 2][16], gen[(iter + 1) % 2][17]);
        to_target_brain_x_crossover(
            gen[iter % 2][idx2], gen[iter % 2][idx4], gen[(iter + 1) % 2][18], gen[(iter + 1) % 2][19]);

        for (auto& e : gen[(iter + 1) % 2])
        {
            if (random::randfloat() < 0.6f)
            {
                e.mutate([] <std::floating_point F> (F f) {
                    const auto r = random::randfloat();
                    if (r < 0.0006f)
                    {
                        return random::randfloat();
                    }
                    if (r < 0.0012f)
                    {
                        return f * random::randnormal();
                    }
                    if (r < 0.0018f)
                    {
                        return decltype(f){};
                    }
                    if (r < 0.0024f)
                    {
                        return f + random::randnormal(random::randnormal(), random::randfloat());
                    }
                    if (r < 0.0030f)
                    {
                        return random::randnormal(f, 1);
                    }
                    return f;
                });
            }
        }

        if (iter % 99 == 0)
        {
            std::cout << iter << " " << *std::ranges::min_element(error) << '\n';
            //output_to_file(iter, best_pred);
        }
    }
}


int main()
{
    training_test();
    // score_function_object_test();
    // in_place_x_crossover_test();

    return EXIT_SUCCESS;
}
