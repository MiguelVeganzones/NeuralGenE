#include "activation_functions.hpp"
#include "evolution_agent.hpp"
#include "neural_model.hpp"
#include "static_neural_net.hpp"
// #include "matplotlibcpp.h"
#include <array>
#include <iostream>
#include <numeric>
#include <ranges>
#include <vector>

// void score_functions_test()
// {
// constexpr size_t N = 10;

// using namespace ga_snn;
// using namespace ga_sm;

// constexpr auto fn = score_functions::score_functions<double, std::uint16_t>::
//     choose_function<score_functions::Identifiers::Weighted_normalized_score, 3, 1, 0>();

// constexpr auto AF   = matrix_activation_functions::Identifiers::GELU;
// constexpr auto Tanh = matrix_activation_functions::Identifiers::Tanh;
// auto           SM =
//     score_function_objects::score_function_object<decltype(fn), std::uint16_t, std::uint16_t, std::uint16_t>(fn);


// using SM_t = decltype(SM);

// constexpr Layer_Signature a1{ 1, AF };
// constexpr Layer_Signature a1_tanh{ 1, Tanh };
// constexpr Layer_Signature a9{ 9, AF };
// constexpr Layer_Signature a25{ 25, AF };

// using NET = static_neural_net<float, 1, a1, a9, a9, a9, a25, a25, a9, a9, a9, a1_tanh>;

// auto brain_a = ga_neural_model::brain<NET, SM_t>(random::randnormal, 0, 1);

// std::array<ga_neural_model::brain<NET, SM_t>, N> gen_a{}, gen_b{};

// for (size_t i = 0; i != N; ++i)
// {
//     gen_a[i] = ga_neural_model::brain<NET, SM_t>(SM, random::randnormal, 0, 1);
//     gen_b[i] = ga_neural_model::brain<NET, SM_t>(SM, random::randnormal, 0, 1);
// }

// std::cout << gen_a[0].get_score_function_obj()->operator()() << std::endl;
// gen_a[0].get_score_function_obj()->lost();
// gen_a[0].get_score_function_obj()->tied();
// gen_a[0].get_score_function_obj()->won();
// std::cout << gen_a[0].get_score_function_obj()->operator()() << std::endl;
// std::cout << gen_a[0].get_score() << std::endl;
// }

void agent_construction()
{
    std::cout << "Evolution agent construction\n";

    random::init();

    // using namespace ga_snn;
    // using namespace ga_sm;

    constexpr auto AF_relu    = matrix_activation_functions::Identifiers::GELU;
    constexpr auto AF_sigmoid = matrix_activation_functions::Identifiers::Sigmoid;
    constexpr auto AF_Tanh    = matrix_activation_functions::Identifiers::Tanh;

    [[maybe_unused]] constexpr ga_snn::Layer_Signature a1{ 1, AF_relu };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a1_Sigmoid{ 1, AF_sigmoid };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a1_Tanh{ 1, AF_Tanh };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a9{ 9, AF_relu };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a25{ 25, AF_relu };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a42{ 42, AF_relu };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a16{ 16, AF_relu };

    using NET = ga_snn::static_neural_net<float, 1, a9, a9, a25, a25, a9, a1_Sigmoid>;
    // using NET = static_neural_net<float, 1, a1, a9, a9, a9, a25, a25, a9, a9, a9, a1_tanh>;

    // std::cout << NET::parameter_count() << std::endl;
    // std::cout << NET2::parameter_count() << std::endl;

    using preprocessor  = data_processor::iterable_converter;
    using postprocessor = data_processor::scalar_converter;
    using return_type   = NET::value_type;

    using brain_t = ga_neural_model::brain<NET, preprocessor, postprocessor, return_type>;
    using agent_t = evolution_agent::agent<brain_t>;


    const auto agent_a = agent_t(std::move(brain_t(random::randnormal, 0, 0.1)), [](auto v) { return v; });
    const auto agent_b = agent_t(std::move(brain_t(random::randnormal, 0, 0.1)), [](auto v) { return v; });

    agent_a.print();
    agent_b.print();

    const auto [agent_c, agent_d] = evolution_agent::crossover(agent_a, agent_b);

    agent_c.print();
    agent_d.print();

    using agent_input_type_1 = std::vector<float>;
    using agent_input_type_2 = std::array<float, 9>;
    using agent_input_type_3 = ga_sm::static_matrix<float, 3, 3>;

    auto input1 = agent_input_type_1(9, 0);
    auto input2 = agent_input_type_2{};
    auto input3 = agent_input_type_3{};

    std::cout << agent_a(input1) << std::endl;
    std::cout << agent_a(input2) << std::endl;
    std::cout << agent_a(input3) << std::endl;

    std::iota(std::begin(input1), std::end(input1), 0);
    std::iota(std::begin(input2), std::end(input2), 0);
    std::iota(std::begin(input3), std::end(input3), 0);

    std::cout << agent_a(input1) << std::endl;
    std::cout << agent_a(input2) << std::endl;
    std::cout << agent_a(input3) << std::endl;


    std::cout << "Here\n" << std::endl;
    std::cout << agent_a(input3) << std::endl;
    std::cout << agent_b(input3) << std::endl;
    std::cout << agent_c(input3) << std::endl;
    std::cout << agent_d(input3) << std::endl;
}

void simple_agent_evolution_test()
{
    random::init();

    // using namespace ga_snn;
    // using namespace ga_sm;

    constexpr auto AF_relu    = matrix_activation_functions::Identifiers::GELU;
    constexpr auto AF_sigmoid = matrix_activation_functions::Identifiers::Sigmoid;
    constexpr auto AF_Tanh    = matrix_activation_functions::Identifiers::Tanh;

    [[maybe_unused]] constexpr ga_snn::Layer_Signature a1{ 1, AF_relu };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a1_Sigmoid{ 1, AF_sigmoid };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a1_Tanh{ 1, AF_Tanh };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a9{ 9, AF_relu };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a25{ 25, AF_relu };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a42{ 42, AF_relu };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a16{ 16, AF_relu };

    using NET = ga_snn::static_neural_net<float, 1, a9, a9, a25, a25, a9, a1_Sigmoid>;
    // using NET = static_neural_net<float, 1, a1, a9, a9, a9, a25, a25, a9, a9, a9, a1_tanh>;

    // std::cout << NET::parameter_count() << std::endl;
    // std::cout << NET2::parameter_count() << std::endl;

    using preprocessor  = data_processor::iterable_converter;
    using postprocessor = data_processor::scalar_converter;
    using return_type   = NET::value_type;

    using brain_t = ga_neural_model::brain<NET, preprocessor, postprocessor, return_type>;
    using agent_t = evolution_agent::agent<brain_t>;

    auto mutation_policy = []<std::floating_point F>(F f) -> F {
        const auto r = random::randfloat();
        if (r >= 0.0009)
        {
            return f * F(0.9999);
        }
        if (r < 0.00006)
        {
            return random::randfloat();
        }
        if (r < 0.00012)
        {
            return f * random::randnormal();
        }
        if (r < 0.00018)
        {
            return F{};
        }
        if (r < 0.00024)
        {
            return f += random::randnormal(random::randnormal(), random::randfloat());
        }
        if (r < 0.00030)
        {
            return random::randnormal(f, F(0.01));
        }
        return f;
    };

    auto agent_a = agent_t(brain_t(random::randnormal, 0, 0.1), mutation_policy);
    auto agent_b = agent_t(brain_t(random::randnormal, 0, 0.1), mutation_policy);

    agent_a.print();
    agent_b.print();

    using agent_input_type = ga_sm::static_matrix<float, 3, 3>;

    agent_input_type input{};
    input.fill(random::randnormal, 0, 1);
    constexpr float target = 0.75;

    for (int i = 0; i != 100000; ++i)
    {
        const auto pred_a  = agent_a(input);
        const auto pred_b  = agent_b(input);
        const auto error_a = std::abs(pred_a - target);
        const auto error_b = std::abs(pred_b - target);

        if (error_a < error_b)
        {
            if (i % 1000 == 0)
            {
                std::cout << "Pred a: " << pred_a << "  (Pred b: " << pred_b << ')' << std::endl;
            }
            agent_b.mutate();
        }
        else if (error_b < error_a)
        {
            if (i % 1000 == 0)
            {
                std::cout << "Pred b: " << pred_b << "  (Pred a: " << pred_a << ')' << std::endl;
            }
            agent_a.mutate();
        }
        if (random::randfloat() < 0.0001)
        {
            evolution_agent::in_place_crossover(agent_a, agent_b);
        }
        // if (i % 5000 == 0)
        // {
        //     agent_a.print();
        // }
    }
}

void multi_agent_evolution_test()
{
    random::init();

    // using namespace ga_snn;
    // using namespace ga_sm;

    [[maybe_unused]] constexpr auto AF_relu    = matrix_activation_functions::Identifiers::ReLU;
    [[maybe_unused]] constexpr auto AF_thresh  = matrix_activation_functions::Identifiers::Threshold;
    [[maybe_unused]] constexpr auto AF_sigmoid = matrix_activation_functions::Identifiers::Sigmoid;
    [[maybe_unused]] constexpr auto AF_Tanh    = matrix_activation_functions::Identifiers::Tanh;

    [[maybe_unused]] constexpr ga_snn::Layer_Signature a1{ 1, AF_relu };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a1_Sigmoid{ 1, AF_sigmoid };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a1_Tanh{ 1, AF_Tanh };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a9{ 9, AF_relu };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a25{ 25, AF_relu };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a42{ 42, AF_relu };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a16{ 16, AF_relu };

    // using NET = ga_snn::static_neural_net<float, 1, a1, a9, a9, a1_Tanh>;
    // using NET = ga_snn::static_neural_net<float, 1, a1, a9, a9, a9, a25, a25, a1_Tanh>;
    using NET = ga_snn::static_neural_net<float, 1, a1, a9, a25, a1_Tanh>;
    // using NET = ga_snn::static_neural_net<float, 1, a1, a9, a9, a9, a9, a25, a9, a9, a25, a1_Tanh>;

    // std::cout << NET::parameter_count() << std::endl;
    // std::cout << NET2::parameter_count() << std::endl;

    using preprocessor  = data_processor::scalar_converter;
    using postprocessor = data_processor::scalar_converter;
    using return_type   = NET::value_type;

    using brain_t = ga_neural_model::brain<NET, preprocessor, postprocessor, return_type>;
    using agent_t = evolution_agent::agent<brain_t>;

    using agent_input_type  = brain_t::value_type;
    using agent_output_type = brain_t::value_type;

    [[maybe_unused]] auto mutation_policy = []<std::floating_point F> [[nodiscard]] (F f) -> F {
        const auto r = random::randfloat();
        if (r >= 0.0009)
        {
            return f * F(0.9999);
        }
        if (r < 0.00006)
        {
            return random::randfloat();
        }
        if (r < 0.00012)
        {
            return f * random::randnormal();
        }
        if (r < 0.00018)
        {
            return F{};
        }
        if (r < 0.00024)
        {
            return f += random::randnormal(random::randnormal(), random::randfloat());
        }
        if (r < 0.00030)
        {
            return random::randnormal(f, F(0.01));
        }
        return f;
    };

    constexpr std::size_t N = 20;
    constexpr std::size_t M = 1000;

    std::vector<std::vector<agent_t>> gen(2);
    for (auto& v : gen)
    {
        v.reserve(N);
    }

    std::vector<float> input(M), output(M), pred(M), best_pred(M);

    for (std::size_t i = 0; i != M; ++i)
    {
        input[i]  = agent_input_type(i);
        output[i] = static_cast<agent_output_type>(std::sin((float)i / 50.f));
    }

    for (size_t i = 0; i != N; ++i)
    {
        gen[0].emplace_back(brain_t(random::randnormal, 0, 0.01), mutation_policy);
        gen[1].emplace_back(brain_t(random::randnormal, 0, 0.01), mutation_policy);
    }

    for (int i = 0; i != 10000; ++i)
    {
        std::array<float, N> error{};
        const int            gen_idx      = i % 2;
        const int            next_gen_idx = (i + 1) % 2;

        for (int jj = 0; jj != N; ++jj)
        {
            for (int ii = 0; ii != M; ++ii)
            {
                pred[ii] = gen[gen_idx][jj](input[ii]);
                error[jj] += std::pow(pred[ii] - output[ii], 2.f);
            }
            if (jj == 0 || (jj > 0 && error[jj] < error[jj - 1]))
            {
                best_pred = pred;
            }
        }

        auto temp = std::array(error);
        std::ranges::sort(temp);

        const auto idx0 = std::ranges::find(error, temp[0]) - error.begin();
        const auto idx1 = std::ranges::find(error, temp[1]) - error.begin();
        const auto idx2 = std::ranges::find(error, temp[2]) - error.begin();
        const auto idx3 = std::ranges::find(error, temp[3]) - error.begin();
        const auto idx4 = std::ranges::find(error, temp[4]) - error.begin();
        const auto idx5 = std::ranges::find(error, temp[5]) - error.begin();

        gen[next_gen_idx][0] = gen[gen_idx][idx0].clone();
        gen[next_gen_idx][1] = gen[gen_idx][idx1].clone();
        // agent_t::to_target_crossover(
        //     gen[gen_idx][idx0], gen[gen_idx][idx1], gen[next_gen_idx][0], gen[next_gen_idx][1]
        // );
        evolution_agent::to_target_crossover(
            gen[gen_idx][idx0], gen[gen_idx][idx2], gen[next_gen_idx][2], gen[next_gen_idx][3]
        );
        evolution_agent::to_target_crossover(
            gen[gen_idx][idx0], gen[gen_idx][idx3], gen[next_gen_idx][4], gen[next_gen_idx][5]
        );
        evolution_agent::to_target_crossover(
            gen[gen_idx][idx0], gen[gen_idx][idx4], gen[next_gen_idx][6], gen[next_gen_idx][7]
        );
        evolution_agent::to_target_crossover(
            gen[gen_idx][idx0], gen[gen_idx][idx5], gen[next_gen_idx][8], gen[next_gen_idx][9]
        );
        evolution_agent::to_target_crossover(
            gen[gen_idx][idx1], gen[gen_idx][idx2], gen[next_gen_idx][10], gen[next_gen_idx][11]
        );
        evolution_agent::to_target_crossover(
            gen[gen_idx][idx1], gen[gen_idx][idx3], gen[next_gen_idx][12], gen[next_gen_idx][13]
        );
        evolution_agent::to_target_crossover(
            gen[gen_idx][idx2], gen[gen_idx][idx3], gen[next_gen_idx][14], gen[next_gen_idx][15]
        );
        evolution_agent::to_target_crossover(
            gen[gen_idx][idx3], gen[gen_idx][idx4], gen[next_gen_idx][16], gen[next_gen_idx][17]
        );
        evolution_agent::to_target_crossover(
            gen[gen_idx][idx4], gen[gen_idx][idx5], gen[next_gen_idx][18], gen[next_gen_idx][19]
        );

        for (auto& e : gen[next_gen_idx])
        {
            if (random::randfloat() < 0.6f)
            {
                e.mutate();
            }
        }
        if (i % 51 == 0)
        {
            std::cout << "Iteration: " << i << ". Min error: " << error[idx0] << std::endl;
        }
    }
}

int main()
{
    std::cout << "Evolution agent main\n";

    // agent_construction();
    // simple_agent_evolution_test();
    multi_agent_evolution_test();

    return EXIT_SUCCESS;
}