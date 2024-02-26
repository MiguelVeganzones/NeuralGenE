#include "Random.hpp"
#include "TApplication.h"
#include "activation_functions.hpp"
#include "data_processor.hpp"
#include "evolution_agent.hpp"
#include "evolution_environment.hpp"
#include "mutation_policy.hpp"
#include "neural_model.hpp"
#include "population.hpp"
#include "reproduction_manager.hpp"
#include "root_plotting_utility.hpp"
#include "static_neural_net.hpp"
#include "system.hpp"
#include <iomanip>
#include <iostream>
#include <ranges>

struct activity
{
    using fitness_score_type  = float;
    using stimulus_type       = float;
    using agent_response_type = float;

    inline static constexpr std::size_t N = 1000;

    using input_data_container  = std::array<stimulus_type, N>;
    using output_data_container = std::array<agent_response_type, N>;

private:
    inline static fitness_score_type s_best_error =
        std::numeric_limits<fitness_score_type>::max();
    inline static std::size_t   s_iter    = 0;
    inline static std::ofstream s_outfile = std::ofstream(
        "./EvolutionEnvironment/Predictions/perf_data.csv",
        std::ios_base::app
    );

public:
    inline static constexpr auto f = [](stimulus_type v
                                     ) -> agent_response_type {
        const auto x = static_cast<float>(v) / 50.f;
        return std::sin(x) * std::cos(std::sqrt(x));
    };

    inline static constexpr auto input_data = []() -> input_data_container {
        input_data_container ret{};
        std::ranges::iota(ret, 0);
        return ret;
    }();

    inline static const auto output_data = []() -> output_data_container {
        output_data_container ret{};
        for (auto i = 0uz; i != N; ++i)
        {
            ret[i] = f(input_data[i]);
        }
        // s_outfile << -1 << ',' << -1 << ',';
        // std::copy(
        // std::begin(ret),
        // std::end(ret),
        // std::ostream_iterator<agent_response_type>(s_outfile, ", ")
        // );
        // s_outfile << '\n';
        return ret;
    }();

    template <typename Agent_Type>
        requires std::
            is_invocable_r_v<agent_response_type, Agent_Type, stimulus_type>
        [[nodiscard]]
        auto
        operator()(Agent_Type&& agent) const -> fitness_score_type
    {
        fitness_score_type fitness_score{};
        for (auto i = 0uz; i != N; ++i)
        {
            fitness_score +=
                static_cast<fitness_score_type>(generics::algorithms::L2_norm(
                    output_data[i], agent(input_data[i])
                ));
        }
        if (/* fitness_score < s_best_error && */ s_iter % 2100 == 0)
        {
            s_best_error = fitness_score;
            // s_outfile << s_best_error << ',';
            // s_outfile << s_iter << ',' << s_best_error << ',';
            // output_data_container pred{};
            // for (auto i = 0uz; i != N; ++i)
            //{
            // pred[i] = agent(input_data[i]);
            //}
            // std::copy(
            // std::begin(pred),
            // std::end(pred),
            // std::ostream_iterator<agent_response_type>(s_outfile, ", ")
            //);
            // s_outfile << '\n';
        }
        ++s_iter;
        return static_cast<fitness_score_type>(1 / fitness_score);
    }

    template <typename Agent_Type>
        requires std::
            is_invocable_r_v<agent_response_type, Agent_Type, stimulus_type>
        auto predict(Agent_Type&& agent) const -> void
    {
        for (auto i = 0uz; i != N; ++i)
        {
            std::cout << input_data[i] << ' ' << output_data[i] << ' '
                      << agent(input_data[i]) << '\n';
        }
    }

    // private:
    //     inline static root_plotting::progress_plot s_plot =
    //         root_plotting::progress_plot(
    //             N,
    //             std::begin(input_data),
    //             std::begin(output_data),
    //             std::begin(output_data)
    //         );
};

int main()

{
    random::init();

    constexpr auto AF_relu = matrix_activation_functions::Identifiers::SiLU;
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
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a32{ 32, AF_relu };
    [[maybe_unused]] constexpr ga_snn::Layer_Signature a64{ 64, AF_relu };

    using NET = // ga_snn::static_neural_net<float, 1, a1, a9, a1_Tanh>;
        ga_snn::static_neural_net<float, 1, a1, a16, a16, a32, a64, a1_Tanh>;

    using preprocessor  = data_processor::scalar_converter;
    using postprocessor = data_processor::scalar_converter;
    using return_type   = typename NET::value_type;

    using brain_t =
        ga_neural_model::brain<NET, preprocessor, postprocessor, return_type>;

    using agent_t = evolution_agent::agent<brain_t>;

    constexpr int GEN_SIZE = 21;
    activity      a;

    [[maybe_unused]] evaluation_system::system<activity> system(a);
    using system_t = decltype(system);

    using mutation_policy_t =
        mutation_policy_::mutation_policy<typename NET::value_type, 6>;


    using reproduction_manager_t = reproduction_mngr::
        reproduction_manager<GEN_SIZE, agent_t, float, mutation_policy_t>;

    using evolution_environment_t = evolution_env::evolution_environment<
        GEN_SIZE,
        agent_t,
        system_t,
        reproduction_manager_t>;

    auto make_agent = []() -> agent_t {
        return agent_t(brain_t(random::randnormal, 0, 0.1));
    };

    reproduction_manager_t reproduction_manager(
        reproduction_mngr::parent_categories(GEN_SIZE, 3, 4)
    );

    [[maybe_unused]] evolution_environment_t eenv(
        make_agent, system, reproduction_manager
    );

    auto [agent, result] = eenv.train(20000);

    a.predict(agent);
    eenv.print_population();

    agent.print();
    std::cout << "Score: " << result << '\n';

    // app.Run();

    return EXIT_SUCCESS;
}