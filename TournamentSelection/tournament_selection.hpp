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

struct tristate_score
{
    int wins;
    int losses;
    int ties;
};

template <std::floating_point F>
class floating_point_normalized_type
{
    F p;

public:
    constexpr explicit floating_point_normalized_type(F f) noexcept :
        p(clamp(f))
    {
    }

    floating_point_normalized_type() noexcept = default;
    floating_point_normalized_type(floating_point_normalized_type const&) noexcept =
        default;
    floating_point_normalized_type(floating_point_normalized_type&&) noexcept =
        default;
    floating_point_normalized_type&
    operator=(floating_point_normalized_type const&) noexcept = default;
    floating_point_normalized_type&
    operator=(floating_point_normalized_type&&) noexcept = default;
    ~floating_point_normalized_type() noexcept           = default;

private:
    inline static constexpr F clamp(F f) noexcept
    {
        return std::clamp(f, F(0), F(1));
    }

public:
    constexpr auto operator+(floating_point_normalized_type const& other
    ) noexcept -> floating_point_normalized_type
    {
        return floating_point_normalized_type(p + other.p);
    }

    constexpr operator F() const noexcept
    {
        return p;
    }

    constexpr floating_point_normalized_type operator/=(F f) noexcept
    {
        assert(f != F(0));
        p = clamp(p / f);
        return *this;
    }

    constexpr floating_point_normalized_type operator+=(F f) noexcept
    {
        p = clamp(p + f);
        return *this;
    }
};

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
concept mutation_policy_generator_concept = requires(T t) {
    typename T::value_type;
    typename T::parameters_type;
    typename T::mutation_policy_type;

    {
        T::default_parameters()
    } -> std::same_as<typename T::parameters_type>;
    {
        T::generate(std::declval<typename T::parameters_type&>())
    } -> std::same_as<typename T::mutation_policy_type>;
    {
        std::declval<typename T::mutation_policy_type&>().get_parameters()
    } -> std::same_as<typename T::parameters_type>;
};

template <
    agent_concept                     Agent_Type,
    std::size_t                       Generation_Size,
    mutation_policy_generator_concept Mutation_Policy_Generator>
    requires(Generation_Size % 2 == 0)
class tournament
{

public:
    inline static constexpr auto s_Generation_size        = Generation_Size;
    inline static constexpr auto s_Population_generations = 2uz;
    inline static constexpr auto s_Population_size =
        s_Generation_size * s_Population_generations;

    inline static auto s_Mutation_probability =
        0.6f; // TODO: Take this argument from the reproduction policy

    using agent_type             = Agent_Type;
    using fitness_score_type     = float;
    using brain_type             = typename agent_type::brain_type;
    using agent_brain_value_type = typename agent_type::brain_value_type;
    using agent_brain_input_type =
        typename agent_type::brain_value_type; // TODO Redo
    using generation_container_type = std::array<agent_type, s_Generation_size>;
    using population_container_type =
        std::array<generation_container_type, s_Population_generations>;

    using mutation_policy_generator_type = Mutation_Policy_Generator;
    using mutation_policy_type =
        typename mutation_policy_generator_type::mutation_policy_type;
    using population_distance_matrix_type =
        ga_sm::static_matrix<R, s_Generation_size, s_Generation_size>;

private:
    inline static auto L2_norm = []<std::floating_point R>(R a, R b) -> R {
        const auto d = (a - b);
        return static_cast<R>(d * d);
    };

    inline static auto L1_norm = []<std::floating_point R>(R a, R b) -> R {
        return static_cast<R>(std::abs(a - b));
    };

private:
    std::size_t                    m_Iterations;
    mutation_policy_generator_type m_Mutation_policy_generator;
    population_container_type      m_Population;


public:
    tournament(
        std::size_t                    iterations,
        mutation_policy_generator_type mutation_policy_generator
    ) :
        m_Iterations{ iterations },
        m_Mutation_policy_generator(mutation_policy_generator),
        m_Population({ make_generation(), make_generation() })
    {
    }

    tournament() noexcept                             = delete;
    tournament(const tournament&) noexcept            = default;
    tournament(tournament&&) noexcept                 = default;
    tournament& operator=(const tournament&) noexcept = default;
    tournament& operator=(tournament&&) noexcept      = default;
    ~tournament() noexcept                            = default;

    [[nodiscard]]
    auto tournament_selection() -> std::pair<agent_type, fitness_score_type>
    {
        constexpr auto M = 1000uz;

        std::vector<float> input(M), output(M);

        for (std::size_t i = 0; i != M; ++i)
        {
            // TODO Redo
            input[i] = agent_brain_input_type(i);
            output[i] =
                static_cast<agent_brain_input_type>(std::sin((float)i / 50.f));
        }
        std::array<float, s_Generation_size> error{};
        constexpr auto                       error_threshold = 50.f;
        for (std::size_t iter = 0; iter != m_Iterations; ++iter)
        {
            const unsigned int gen_idx = iter % s_Population_generations;
            const unsigned int next_gen_idx =
                (iter + 1) % s_Population_generations;

            /// -------------------------------------
            for (std::size_t j = 0; j != s_Generation_size; ++j)
            {
                error[j] = 0;
                for (std::size_t i = 0; i != M; ++i)
                {
                    error[j] +=
                        L2_norm(m_Population[gen_idx][j](input[i]), output[i]);
                }
                if (error[j] < error_threshold)
                {
                    std::cout << "Error: " << error[j] << " at iter: " << iter
                              << '\n';
                    for (auto i = 0u; i != M; ++i)
                    {
                        std::cout << m_Population[gen_idx][j](input[i])
                                  << " :: " << output[i] << "\n";
                    }
                    return { m_Population[gen_idx][j], error[j] };
                }
            }
            const auto max_error = std::ranges::max(error);
            const auto min_error = std::ranges::min(error);
            std::cout << min_error << " - " << max_error << std::endl;


            scores_container_type agent_fitness{};
            std::transform(
                error.cbegin(),
                error.cend(),
                agent_fitness.begin(),
                [max_error](float error_) {
                    return float_normalized_type(1 - error_ / max_error);
                }
            );

            /// -------------------------------------

            [[maybe_unused]] const auto selected_parents = select_parents(
                generation_variability<float>(m_Population[gen_idx], L2_norm),
                agent_fitness
            );

            if (iter == m_Iterations - 1)
            {
                return { m_Population[gen_idx][0], error[0] };
            }

            for (std::size_t i = 0;
                 auto [parent_a, parent_b] : selected_parents)
            {
                to_target_crossover(
                    m_Population[gen_idx][parent_a],
                    m_Population[gen_idx][parent_b],
                    m_Population[next_gen_idx][i],
                    m_Population[next_gen_idx][i + 1]
                );
                i += 2;
            }

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
    template <std::floating_point R, typename Distance>
    [[nodiscard]]
    static auto generation_variability(
        std::array<agent_type, s_Generation_size> const& agents,
        Distance&&                                       dist_op
    ) -> population_distance_matrix_type
    {
        population_distance_matrix_type distance_matrix{};
        for (size_t j = 0; j != s_Generation_size - 1; ++j)
        {
            for (size_t i = j + 1; i != s_Generation_size; ++i)
            {
                const auto distance = agent_distance<R>(
                    agents[j], agents[i], std::forward<Distance>(dist_op)
                );
                distance_matrix[j, i] = distance;
                distance_matrix[i, j] = distance;
            }
        }
        return distance_matrix;
    }

    template <size_t... Is>
    [[nodiscard]]
    auto generation_factory(std::index_sequence<Is...>)
        -> std::array<agent_type, sizeof...(Is)>
    {
        return { ((void)Is, make_agent())... };
    }

    [[nodiscard]]
    auto make_generation() -> generation_container_type
    {
        return generation_factory(std::make_index_sequence<s_Generation_size>()
        );
    }

    [[nodiscard]]
    auto make_agent() const -> agent_type
    {
        return agent_type(
            brain_type(random::randnormal, 0, 0.5),
            m_Mutation_policy_generator.generate(
                mutation_policy_generator_type::default_parameters()
            )
        );
    }

} // namespace tournament_selection

#endif // TOURNAMENT_SELECTION
