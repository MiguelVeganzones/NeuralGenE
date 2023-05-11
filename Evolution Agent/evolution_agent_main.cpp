#include "evolution_agent.hpp"
#include <array>
#include <iostream>
#include <numeric>
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

int main()
{
    std::cout << "Evolution agent main\n";

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

    using brain_t     = ga_neural_model::brain<NET, preprocessor, postprocessor, return_type>;
    constexpr auto fn = score_functions::score_functions<double, std::uint16_t>::
        choose_function<score_functions::Identifiers::Weighted_normalized_score, 3, 1, 0>();
    using score_function_t = score_function_objects::score_function_object<decltype(fn), float, float, float>;

    using agent_t = evolution_agent::agent<brain_t, score_function_t>;


    auto agent_a = agent_t(std::move(brain_t(random::randnormal, 0, 0.1)), score_function_t(fn));
    auto agent_b = agent_t(std::move(brain_t(random::randnormal, 0, 0.1)), score_function_t(fn));

    agent_a.print();
    agent_b.print();

    auto [agent_c, agent_d] = agent_t::x_crossover(agent_a, agent_b);

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


    return EXIT_SUCCESS;
}