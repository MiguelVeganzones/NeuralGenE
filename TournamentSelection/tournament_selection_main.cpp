#include "Random.hpp"
#include "activation_functions.hpp"
#include "data_processor.hpp"
#include "evolution_agent.hpp"
#include "fitness_calculator.hpp"
#include "mutation_policy.hpp"
#include "neural_model.hpp"
#include "static_neural_net.hpp"
#include "tournament_selection.hpp"
#include <array>
#include <iostream>

int main()
{
    random::seed();

    constexpr auto AF_relu = matrix_activation_functions::Identifiers::GELU;
    constexpr auto AF_sigmoid =
        matrix_activation_functions::Identifiers::Sigmoid;
    constexpr auto AF_Tanh = matrix_activation_functions::Identifiers::Tanh;

    [[maybe_unused]] constexpr ga_snn::Layer_Signature a1{ 1, AF_relu };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a1_Sigmoid{ 1,
                                                                   AF_sigmoid };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a1_Tanh{ 1, AF_Tanh };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a9{ 9, AF_relu };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a25{ 25, AF_relu };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a42{ 42, AF_relu };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a16{ 16, AF_relu };

    using NET =
        ga_snn::static_neural_net<float, 1, a1, a9, a9, a9, a25, a25, a1_Tanh>;
    // using NET = ga_snn::static_neural_net<float, 1, a1, a9, a9, a9, a25, a25,
    // a1_Tanh>;

    // std::cout << NET::parameter_count() << std::endl;
    // std::cout << NET2::parameter_count() << std::endl;

    using preprocessor  = data_processor::scalar_converter;
    using postprocessor = data_processor::scalar_converter;
    using return_type   = NET::value_type;

    using brain_t =
        ga_neural_model::brain<NET, preprocessor, postprocessor, return_type>;


    using mutation_policy_generator_t =
        mutation_policy::mutation_policy_generator<NET::value_type>;
    using agent_t = evolution_agent::
        agent<brain_t, mutation_policy_generator_t::mutation_policy_type>;

    using tournament_t = tournament_selection::
        tournament<agent_t, 20, mutation_policy_generator_t>;

    auto tournament = tournament_t(100000, mutation_policy_generator_t());

    auto [winner_agent, error] = tournament.tournament_selection();
    winner_agent.print();
    std::cout << "Error: " << error << '\n';

    return EXIT_SUCCESS;
}