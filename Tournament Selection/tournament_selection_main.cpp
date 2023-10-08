#include "Random.h"
#include "activation_functions.hpp"
#include "data_processor.hpp"
#include "evolution_agent.hpp"
#include "neural_model.hpp"
#include "static_neural_net.hpp"
#include "tournament_selection.hpp"
#include <array>
#include <iostream>

template <std::floating_point Value_Type>
class mutation_policy_generator
{
public:
    inline static constexpr std::size_t N = 6;

    using value_type             = Value_Type;
    using probability_type       = tournament_selection::floating_point_normalized_type<float>;
    using parameters_type        = std::array<probability_type, N>;
    using mutation_function_type = value_type (*)(value_type);

    class mutation_policy_type
    {
    public:
        using value_type      = mutation_policy_generator::value_type;
        using parameters_type = mutation_policy_generator::parameters_type;

    private:
        parameters_type m_Params;
        parameters_type m_Cummulative_params;

    public:
        mutation_policy_type(parameters_type const& params) noexcept :
            m_Params(params)
        {
            for (int i = 1; i != N; ++i)
            {
                m_Cummulative_params[i] = m_Params[i] + m_Cummulative_params[i - 1];
            }
        }

        mutation_policy_type() noexcept                                       = delete;
        mutation_policy_type(mutation_policy_type const&) noexcept            = default;
        mutation_policy_type(mutation_policy_type&&) noexcept                 = default;
        mutation_policy_type& operator=(mutation_policy_type const&) noexcept = default;
        mutation_policy_type& operator=(mutation_policy_type&&) noexcept      = default;
        ~mutation_policy_type() noexcept                                      = default;

        [[nodiscard]] auto get_parameters() const noexcept -> parameters_type
        {
            return m_Params;
        }

        [[nodiscard]] auto operator()(value_type f) const noexcept -> value_type
        {
            const auto r = random::randfloat();
            if (r >= m_Cummulative_params[5])
            {
                return f;
            }
            if (r < m_Cummulative_params[0])
            {
                return random::randfloat();
            }
            if (r < m_Cummulative_params[1])
            {
                return f * random::randnormal();
            }
            if (r < m_Cummulative_params[2])
            {
                return value_type{};
            }
            if (r < m_Cummulative_params[3])
            {
                return f += random::randnormal(random::randnormal(), random::randfloat());
            }
            if (r < m_Cummulative_params[4])
            {
                return random::randnormal(f, value_type(0.01));
            }
            if (r < m_Cummulative_params[5])
            {
                return f * value_type(0.9999);
            }
            assert_unreachable();
        }
    };

    [[nodiscard]] static auto default_parameters() noexcept -> parameters_type
    {
        return s_Default_params;
    }

    inline static constexpr probability_type p0               = probability_type(0.0005f);
    inline static constexpr parameters_type  s_Default_params = { p0, p0, p0, p0, p0, p0 };

public:
    [[nodiscard]] static auto generate(parameters_type const& params) noexcept -> mutation_policy_type
    {
        // TODO mutate params
        return mutation_policy_type(params);
    }
};

// TODO Restrict template
// template <
//     typename generation_container_type,
//     typename scores_container_type,
//     typename reproduction_probabilities_container_type>
template <typename Agent>
struct fitness_calculator
{
    using agent_type       = Agent; // std::remove_cvref_t<typename generation_container_type::value_type>;
    using probability_type = tournament_selection::floating_point_normalized_type<float>;

    inline static constexpr float distance_threshold = 0.1f; // TODO

    // auto operator()(generation_container_type const&, scores_container_type const&)
    // {
    //     reproduction_probabilities_container_type();
    // }

    template <std::floating_point R, size_t N>
        requires(N > 1)
    [[nodiscard]] static auto generation_variability(std::array<agent_type, N> const& agents)
        -> ga_sm::static_matrix<R, N, N>
    {
        ga_sm::static_matrix<R, N, N> distance_matrix{};
        for (size_t j = 0; j != N - 1; ++j)
        {
            for (size_t i = j + 1; i != N; ++i)
            {
                const auto _distance =
                    distance<R>(agents[j], agents[i], [](float a, float b) { return std::pow(a - b, 2); });
                distance_matrix[j, i] = _distance;
                distance_matrix[i, j] = _distance;
            }
        }
        return distance_matrix;
    }

    template <std::size_t N>
        requires(N > 1)
    //[[nodiscard]]
    static auto generation_fitness(std::array<agent_type, N> const& agents, std::array<float, N> generation_loss)
        -> std::array<probability_type, N>
    {
        const auto distance_matrix = generation_variability(agents);

        std::array<std::size_t, N> indexer{};
        std::iota(indexer.begin(), indexer.end(), 0);

        std::sort(
            indexer.begin(),
            indexer.end(),
            [&generation_loss](const std::size_t idx_1, const std::size_t idx_2) -> bool {
                return generation_loss[idx_1] < generation_loss[idx_2];
            }
        );

        std::array<probability_type, N> scores{};
        std::array<bool, N>             done{ false };
        const auto                      max_loss = std::ranges::max(generation_loss);

        for (auto j : indexer)
        {
            if (done[j])
                continue;
            scores[j] = 1 - (generation_loss[j] / max_loss);
            for (std::size_t i = 0; i != N; ++i)
            {
                if (i == j || done[i])
                {
                    continue;
                }
                if (distance_matrix[j, i] < distance_threshold)
                {
                    done[i] = true;
                    continue;
                }
            }
        }
        for (auto e : scores)
            std::cout << e << ' ';
        std::cout << std::endl;
        return scores;
    }
};

int main()
{
    random::init();

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

    using NET = ga_snn::static_neural_net<float, 1, a1, a9, a9, a9, a25, a25, a1_Tanh>;
    // using NET = ga_snn::static_neural_net<float, 1, a1, a9, a9, a9, a25, a25, a1_Tanh>;

    // std::cout << NET::parameter_count() << std::endl;
    // std::cout << NET2::parameter_count() << std::endl;

    using preprocessor  = data_processor::scalar_converter;
    using postprocessor = data_processor::scalar_converter;
    using return_type   = NET::value_type;

    using brain_t = ga_neural_model::brain<NET, preprocessor, postprocessor, return_type>;


    using mutation_policy_generator_t = mutation_policy_generator<NET::value_type>;
    using agent_t = evolution_agent::agent<brain_t, mutation_policy_generator_t::mutation_policy_type>;

    using tournament_t = tournament_selection::tournament<agent_t, 20, mutation_policy_generator_t>;

    auto tournament = tournament_t(100000, mutation_policy_generator_t());

    auto [winner_agent, error] = tournament.tournament_selection();
    winner_agent.print();
    std::cout << "Error: " << error << '\n';

    return EXIT_SUCCESS;
}