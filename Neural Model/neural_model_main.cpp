#include "Random.h"
#include "Stopwatch.h"
#include "activation_functions.hpp"
#include "neural_model.hpp"
#include "score_functions.hpp"
#include "static_matrix.hpp"
#include "static_neural_net.hpp"


// void in_place_x_crossover_test()
//{
//     using namespace ga_snn;
//     using namespace ga_sm;
//
//     constexpr auto AF = matrix_activation_functions::Identifiers::ReLU;
//     constexpr auto SM =
//     ga_neural_model::score_function_params{ score_functions::Identifiers::Weighted_normalized_score, 3, 1, 0 };
//
//     constexpr Layer_Signature a1{ 1, AF };
//     constexpr Layer_Signature a9{ 9, AF };
//     constexpr Layer_Signature a25{ 25, AF };
//
//     using NET = static_neural_net<float, 1, a1, a9, a25, a25, a25, a25, a9, a1>;
//
//     auto brain_a = ga_neural_model::brain<NET, SM>(random::randnormal, 0, 1);
//     auto brain_b = ga_neural_model::brain<NET, SM>(random::randnormal, 0, 1);
//     auto brain_c = ga_neural_model::brain<NET, SM>(random::randnormal, 0, 1);
//     auto brain_d = ga_neural_model::brain<NET, SM>(random::randnormal, 0, 1);
//
//     std::vector gen1 = { std::move(brain_a), std::move(brain_b) };
//     std::vector gen2 = { std::move(brain_c), std::move(brain_d) };
//
//     gen1[0].print_net_address();
//     gen1[1].print_net_address();
//     gen2[0].print_net_address();
//     gen2[1].print_net_address();
//
//     std::cout << "--------------\n";
//
//     {
//         stopwatch s;
//         auto [b1, b2] = brain_x_crossover(gen1[0], gen1[1]);
//     }
//     {
//         stopwatch s;
//         to_target_brain_x_crossover(gen1[0], gen1[1], gen2[0], gen2[1]);
//     }
//     to_target_brain_x_crossover(gen1[0], gen1[1], gen2[0], gen2[1]);
//
//     gen1[0].print_net_address();
//     gen1[1].print_net_address();
//     gen2[0].print_net_address();
//     gen2[1].print_net_address();
//
//     std::cout << "--------------\n";
//
//     to_target_brain_x_crossover(gen1[0], gen1[1], gen2[0], gen2[1]);
//     to_target_brain_x_crossover(gen1[0], gen1[1], gen2[0], gen2[1]);
//
//     gen1[0].print_net_address();
//     gen1[1].print_net_address();
//     gen2[0].print_net_address();
//     gen2[1].print_net_address();
//
//     std::cout << "--------------\n";
//
//     constexpr mutate_params params{ 0.2, 0.2, 0, 1 };
//
//     gen1[0].print_net();
//
//     gen1[0].mutate_set_layers({ 0, 2, 4, 6 }, params, random::randfloat);
//
//     gen1[0].print_net();
// }

void training_test0()
{
    constexpr size_t N = 10;

    using namespace ga_snn;
    using namespace ga_sm;

    constexpr auto AF = matrix_activation_functions::Identifiers::ReLU;
    constexpr auto SM = ga_neural_model::score_function_object(
        score_functions::score_functions<double, int>::
            choose_function<score_functions::Identifiers::Weighted_normalized_score, 3, 1, 0>(),
        0,
        0,
        0);

    using SM_t = decltype(SM);

    constexpr Layer_Signature a1{ 1, AF };
    constexpr Layer_Signature a9{ 9, AF };
    constexpr Layer_Signature a25{ 25, AF };

    using NET = static_neural_net<float, 1, a1, a9, a25, a25, a25, a25, a9, a1>;

    auto brain_a = ga_neural_model::brain<NET, SM_t>(random::randnormal, 0, 1);

    std::array<ga_neural_model::brain<NET, SM_t>, N> gen_a{}, gen_b{};

    for (size_t i = 0; i != N; ++i)
    {
        gen_a[i] = ga_neural_model::brain<NET, SM_t>(random::randnormal, 0, 1);
        gen_b[i] = ga_neural_model::brain<NET, SM_t>(random::randnormal, 0, 1);
    }
}


void score_function_object_test()
{
    auto f = [](std::tuple<int, int> const& t) { return std::get<0>(t) + std::get<1>(t); };
    ga_neural_model::score_function_object f_obj(f, 0, 1);

    std::cout << f_obj() << std::endl;
}

int main()
{
    score_function_object_test();
    // in_place_x_crossover_test();
    return EXIT_SUCCESS;
}
