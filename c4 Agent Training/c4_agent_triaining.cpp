#include "Random.h"
#include "activation_functions.hpp"
#include "data_processor.hpp"
#include "evolution_agent.hpp"
#include "neural_model.hpp"
#include "score_functions.hpp"
#include "static_matrix.hpp"
#include "static_neural_net.hpp"

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

    using brain_t     = ga_neural_model::brain<NET, preprocessor, postprocessor, return_type>;
    constexpr auto fn = score_functions::score_functions<float, std::uint16_t>::
        choose_function<score_functions::Identifiers::Weighted_normalized_score, 3, 1, 0>();
    using score_function_t = score_function_objects::score_function_object<decltype(fn), float, float, float>;

    using agent_t = evolution_agent::agent<brain_t, score_function_t>;

    using agent_input_type  = brain_t::value_type;
    using agent_output_type = brain_t::value_type;

    [[maybe_unused]] auto mutation_policy = []<std::floating_point F>(F f) -> F {
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

    std::array<std::array<agent_t, N>, 2> gen{};

    std::vector<float> input(M), output(M), pred(M), best_pred(M);

    for (std::size_t i = 0; i != M; ++i)
    {
        input[i]  = agent_input_type(i);
        output[i] = static_cast<agent_output_type>(std::sin((float)i / 50.f));
    }

    for (size_t i = 0; i != N; ++i)
    {
        gen[0][i] = agent_t(brain_t(random::randnormal, 0, 0.01), score_function_t(fn));
        gen[1][i] = agent_t(brain_t(random::randnormal, 0, 0.01), score_function_t(fn));
    }

    for (int i = 0; i != 100000; ++i)
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
        agent_t::to_target_crossover(
            gen[gen_idx][idx0], gen[gen_idx][idx2], gen[next_gen_idx][2], gen[next_gen_idx][3]
        );
        agent_t::to_target_crossover(
            gen[gen_idx][idx0], gen[gen_idx][idx3], gen[next_gen_idx][4], gen[next_gen_idx][5]
        );
        agent_t::to_target_crossover(
            gen[gen_idx][idx0], gen[gen_idx][idx4], gen[next_gen_idx][6], gen[next_gen_idx][7]
        );
        agent_t::to_target_crossover(
            gen[gen_idx][idx0], gen[gen_idx][idx5], gen[next_gen_idx][8], gen[next_gen_idx][9]
        );
        agent_t::to_target_crossover(
            gen[gen_idx][idx1], gen[gen_idx][idx2], gen[next_gen_idx][10], gen[next_gen_idx][11]
        );
        agent_t::to_target_crossover(
            gen[gen_idx][idx1], gen[gen_idx][idx3], gen[next_gen_idx][12], gen[next_gen_idx][13]
        );
        agent_t::to_target_crossover(
            gen[gen_idx][idx2], gen[gen_idx][idx3], gen[next_gen_idx][14], gen[next_gen_idx][15]
        );
        agent_t::to_target_crossover(
            gen[gen_idx][idx3], gen[gen_idx][idx4], gen[next_gen_idx][16], gen[next_gen_idx][17]
        );
        agent_t::to_target_crossover(
            gen[gen_idx][idx4], gen[gen_idx][idx5], gen[next_gen_idx][18], gen[next_gen_idx][19]
        );

        for (auto& e : gen[next_gen_idx])
        {
            if (random::randfloat() < 0.6f)
            {
                e.mutate(mutation_policy);
            }
        }
        if (i % 51 == 0)
        {
            std::cout << "Iteration: " << i << ". Min error: " << error[idx0] << std::endl;
        }
    }
}
