#ifndef TOURNAMENT_SELECTION
#define TOURNAMENT_SELECTION

#include "Random.hpp"
#include "activation_functions.hpp"
#include "error_handling.hpp"
#include "static_matrix.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <type_traits>
#include <vector>

namespace tournament_selection
{

template <typename T>
concept agent_concept = requires(T t) {
    typename T::brain_value_type;
    {
        t.mutate()
    } -> std::same_as<void>;
    {
        to_target_crossover(t, t, t, t)
    } -> std::same_as<void>;
};

template <typename T>
concept reproduction_policy_concept = requires(T t) {
    typename T::chosen_parent_type;
    typename T::chosen_parents_container_type;
    T::Population_Size;
    {
        t.select_parents(ga_sm::static_matrix<float, T::Population_Size, T::Population_Size>, std::array<float, T::Population_Size>)
    } -> std::same_as<typename T::chosen_parents_container_type>;
};

template <
    agent_concept                     Agent_Type,
    std::uint16_t                     Generation_Size,
    reproduction_policy_concept       Reproduction_Policy,
    mutation_policy_generator_concept Mutation_Policy_Generator>
class tournament
{
public:
    inline static constexpr auto s_Generation_size        = Generation_Size;
    inline static constexpr auto s_Population_generations = 2uz;
    inline static constexpr auto s_Population_size        = s_Generation_size * s_Population_generations;

    using agent_type                = Agent_Type;
    using fitness_score_type        = float;
    using brain_type                = typename agent_type::brain_type;
    using agent_brain_value_type    = typename agent_type::brain_value_type;
    using agent_brain_input_type    = typename agent_type::brain_value_type; // TODO Redo
    using generation_container_type = std::array<agent_type, s_Generation_size>;
    using population_container_type = std::array<generation_container_type, s_Population_generations>;
    using fitness_container_type    = std::array<float, s_Generation_size>;

    using reproduction_policy_type       = Reproduction_Policy;
    using mutation_policy_generator_type = Mutation_Policy_Generator;
    using mutation_policy_type           = typename mutation_policy_generator_type::mutation_policy_type;

public:
    tournament(
        std::size_t                    iterations,
        reproduction_policy_type       reproduction_policy,
        mutation_policy_generator_type mutation_policy_generator
    ) :
        m_Iterations{ iterations },
        m_Reproduction_policy{ reproduction_policy },
        m_Mutation_policy_generator(mutation_policy_generator),
        m_Population({ make_generation(), make_generation() })
    {
    }

    tournament() noexcept                             = delete;
    tournament(const tournament&) noexcept            = delete;
    tournament(tournament&&) noexcept                 = delete;
    tournament& operator=(const tournament&) noexcept = delete;
    tournament& operator=(tournament&&) noexcept      = default;
    ~tournament() noexcept;

private:
    inline static auto L2_norm = []<std::floating_point R>(R a, R b) -> R {
        const auto d = (a - b);
        return static_cast<R>(d * d);
    };

    inline static auto L1_norm = []<std::floating_point R>(R a, R b) -> R { return static_cast<R>(std::abs(a - b)); };

    [[nodiscard]]
    auto make_generation() -> generation_container_type
    {
        return generation_factory(std::make_index_sequence<s_Generation_size>());
    }

    template <size_t... Is>
    [[nodiscard]]
    auto generation_factory(std::index_sequence<Is...>) -> std::array<agent_type, sizeof...(Is)>
    {
        return { ((void)Is, make_agent())... };
    }

    [[nodiscard]]
    auto make_agent() const -> agent_type
    {
        return agent_type(
            brain_type(random::randnormal, 0, 0.5),
            m_Mutation_policy_generator.generate(mutation_policy_generator_type::default_parameters())
        );
    }

public:
    [[nodiscard]]
    auto tournament_selection() -> std::pair<agent_type, fitness_score_type>
    {
        constexpr auto                      M = 1000uz;
        std::vector<agent_brain_input_type> input(M), output(M);

        for (std::size_t i = 0; i != M; ++i)
        {
            // TODO Redo
            input[i]  = agent_brain_input_type(i);
            output[i] = static_cast<agent_brain_input_type>(std::sin((float)i / 50.f));
        }
        std::array<float, s_Generation_size> error{};
        constexpr auto                       error_threshold = 50.f;

        for (auto iter = 0uz; iter != m_Iterations; ++iter)
        {
            const unsigned int gen_idx      = iter % s_Population_generations;
            const unsigned int next_gen_idx = (iter + 1) % s_Population_generations;

            /// -------------------------------------
            for (std::size_t j = 0; j != s_Generation_size; ++j)
            {
                error[j] = 0;
                for (std::size_t i = 0; i != M; ++i)
                {
                    error[j] += L2_norm(m_Population[gen_idx][j](input[i]), output[i]);
                }
                if (error[j] < error_threshold)
                {
                    std::cout << "Error: " << error[j] << " at iter: " << iter << '\n';
                    for (auto i = 0u; i != M; ++i)
                    {
                        std::cout << m_Population[gen_idx][j](input[i]) << " :: " << output[i] << "\n";
                    }
                    return { m_Population[gen_idx][j], error[j] };
                }
            }
            const auto max_error = std::ranges::max(error);
            const auto min_error = std::ranges::min(error);
            std::cout << min_error << " - " << max_error << std::endl;

            fitness_container_type agent_fitness{};
            std::transform(error.cbegin(), error.cend(), agent_fitness.begin(), [](float error_) { return -error_; });

            /// -------------------------------------

            const auto selected_parents = m_Reproduction_policy.select_parents(
                generation_variability<float>(m_Population[gen_idx], L2_norm), agent_fitness
            );

            if (iter == m_Iterations - 1)
            {
                return { m_Population[gen_idx][0], error[0] };
            }

            m_Reproduction_policy.reproduce(m_Population[gen_idx], m_Population[next_gen_idx], selected_parents);

            for (auto& agent : m_Population[next_gen_idx])
            {
                if (random::randfloat() < s_Mutation_probability)
                {
                    agent.mutate();
                }
            }
        }
        assert_unreachable();
    }


private:
    std::size_t                    m_Iterations;
    reproduction_policy_type       m_Reproduction_policy;
    mutation_policy_generator_type m_Mutation_policy_generator;
    population_container_type      m_Population;
};

} // namespace tournament_selection

#endif // TOURNAMENT_SELECTION
