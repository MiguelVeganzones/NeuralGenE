#include "Random.h"
#include "activation_functions.hpp"
#include "data_processor.hpp"
#include "evolution_agent.hpp"
#include "neural_model.hpp"
#include "score_functions.hpp"
#include "static_matrix.hpp"
#include "static_neural_net.hpp"
#include <iostream>

void multi_agent_evolution_test()
{
    random::init();

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

    using agent_input_type            = brain_t::value_type;
    using agent_output_type           = brain_t::value_type;
    constexpr int                  P  = 5;
    constexpr float                p0 = 0.00006f;
    constexpr std::array<float, P> mutation_probabilities{ p0, p0, p0, p0, p0 };
    std::array<float, P>           cummulative_mutation_probabilities{};

    for (int i = 0; i != P; ++i)
    {
        cummulative_mutation_probabilities[i] =
            mutation_probabilities[i] + (i > 0 ? cummulative_mutation_probabilities[i - 1] : 0);
    }
    for (auto e : cummulative_mutation_probabilities)
    {
        std::cout << e << std::endl;
    }

    auto mutation_policy = [&]<std::floating_point F>(F f) -> F {
        const auto r = random::randfloat();
        if (r >= cummulative_mutation_probabilities[4] * 2)
        {
            return f * F(0.9999);
        }
        if (r >= cummulative_mutation_probabilities[4])
        {
            return f;
        }
        if (r < cummulative_mutation_probabilities[0])
        {
            return random::randfloat();
        }
        if (r < cummulative_mutation_probabilities[1])
        {
            return f * (F(1) + random::randnormal(0, 0.001f));
        }
        if (r < cummulative_mutation_probabilities[2])
        {
            return F();
        }
        if (r < cummulative_mutation_probabilities[3])
        {
            return f += random::randnormal(random::randnormal(), random::randfloat() * 0.1f);
        }
        if (r < cummulative_mutation_probabilities[4])
        {
            return random::randnormal(f, F(0.01));
        }
        return f;
    };

    constexpr std::size_t N = 20;

    std::array<std::array<agent_t, N>, 2> gen{};

    for (size_t i = 0; i != N; ++i)
    {
        gen[0][i] = agent_t(brain_t(random::randnormal, 0, 0.001f));
        gen[1][i] = agent_t(brain_t(random::randnormal, 0, 0.001f));
    }

    for (int i = 0; i != 100000; ++i)
    {
        const int gen_idx      = i % 2;
        const int next_gen_idx = (i + 1) % 2;

        for (auto& e : gen[next_gen_idx])
        {
            if (random::randfloat() < 0.6f)
            {
                e.mutate(mutation_policy);
            }
        }
    }
}

int main()
{
    multi_agent_evolution_test();
}