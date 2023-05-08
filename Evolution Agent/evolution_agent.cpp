#include "evolution_agent.hpp"
#include <iostream>

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
    return EXIT_SUCCESS;
}