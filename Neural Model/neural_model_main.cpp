#include "Random.h"
#include "activation_functions.hpp"
#include "neural_model.hpp"
#include "static_matrix.hpp"
#include "static_neural_net.hpp"
#include "score_functions.hpp"


void in_place_x_crossover_test()
{
    using namespace ga_snn;
    using namespace ga_sm;

    constexpr auto AF = matrix_activation_functions::Identifiers::ReLU;
    constexpr auto SM =
        ga_neural_model::score_function_params{ score_functions::Identifiers::Weighted_normalized_score, 3, 1, 0 };

    constexpr Layer_Signature a1{ 1, AF };
    // constexpr Layer_Signature a2{ 4, AF };
    constexpr Layer_Signature a9{ 9, AF };
    // constexpr Layer_Signature a4{ 16, AF };
    // constexpr Layer_Signature a25{ 25, AF };

    using NET = static_neural_net<float, 1, a1, a9, a1>;

    auto brain_a = ga_neural_model::brain<NET, SM>(random::randfloat);
    auto brain_b = ga_neural_model::brain<NET, SM>(random::randfloat);

    ga_neural_model::brain<NET, SM> brain_c, brain_d;

    brain_a.print_net();
    brain_b.print_net();

    to_target_brain_x_crossover(brain_a, brain_b, brain_c, brain_d);

    brain_c.print_net();
    brain_d.print_net();

    std::cout << brain_a.get_Score() << std::endl;
}

int main()
{
    in_place_x_crossover_test();
    return EXIT_SUCCESS;
}
