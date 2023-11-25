#ifndef TOURNAMENT_SELECTION
#define TOURNAMENT_SELECTION

#include "Random.h"
#include "activation_functions.hpp"
#include "error_handling.h"
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

    floating_point_normalized_type() noexcept                                                 = default;
    floating_point_normalized_type(floating_point_normalized_type const&) noexcept            = default;
    floating_point_normalized_type(floating_point_normalized_type&&) noexcept                 = default;
    floating_point_normalized_type& operator=(floating_point_normalized_type const&) noexcept = default;
    floating_point_normalized_type& operator=(floating_point_normalized_type&&) noexcept      = default;
    ~floating_point_normalized_type() noexcept                                                = default;

private:
    inline static constexpr F clamp(F f) noexcept
    {
        return std::clamp(f, F(0), F(1));
    }

public:
    constexpr auto operator+(floating_point_normalized_type const& other) noexcept -> floating_point_normalized_type
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
                                                    T::generate(std::declval<typename T::parameters_type>())
                                                    } -> std::same_as<typename T::mutation_policy_type>;
                                                {
                                                    std::declval<typename T::mutation_policy_type>().get_parameters()
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
    inline static constexpr auto s_Population_size        = s_Generation_size * s_Population_generations;

    inline static auto s_Mutation_probability = 0.6f; // TODO: Take this argument from the reproduction policy

    using agent_type                = Agent_Type;
    using fitness_score_type        = float;
    using brain_type                = typename agent_type::brain_type;
    using agent_brain_value_type    = typename agent_type::brain_value_type;
    using agent_brain_input_type    = typename agent_type::brain_value_type; // TODO Redo
    using generation_container_type = std::array<agent_type, s_Generation_size>;
    using population_container_type = std::array<generation_container_type, s_Population_generations>;
    using float_normalized_type     = floating_point_normalized_type<float>;
    template <std::size_t N>
    using float_normalized_array        = std::array<float_normalized_type, N>;
    using parent_fitness_container_type = float_normalized_array<s_Generation_size>;

    using score_type          = tristate_score;
    using score_function_type = fitness_score_type (*)(score_type const&);

    using mutation_policy_generator_type = Mutation_Policy_Generator;
    using scores_container_type =
        std::array<float_normalized_type, s_Generation_size>; // typename
                                                              // mutation_policy_generator_type::fitness_container_type;
    using mutation_policy_type = typename mutation_policy_generator_type::mutation_policy_type;

private:
    inline static auto L2_norm = []<std::floating_point R>(R a, R b) -> R {
        const auto d = (a - b);
        return static_cast<R>(d * d);
    };

    inline static auto L1_norm = []<std::floating_point R>(R a, R b) -> R { return static_cast<R>(std::abs(a - b)); };

private:
    std::size_t                    m_Iterations;
    mutation_policy_generator_type m_Mutation_policy_generator;
    population_container_type      m_Population;


public:
    tournament(std::size_t iterations, mutation_policy_generator_type mutation_policy_generator) :
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

    [[nodiscard]] auto tournament_selection() -> std::pair<agent_type, fitness_score_type>
    {
        constexpr auto M = 1000uz;

        std::vector<float> input(M), output(M);

        for (std::size_t i = 0; i != M; ++i)
        {
            // TODO Redo
            input[i]  = agent_brain_input_type(i);
            output[i] = static_cast<agent_brain_input_type>(std::sin((float)i / 50.f));
        }
        std::array<float, s_Generation_size> error{};
        constexpr auto                       error_threshold = 50.f;
        for (std::size_t iter = 0; iter != m_Iterations; ++iter)
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


            scores_container_type agent_fitness{};
            std::transform(error.cbegin(), error.cend(), agent_fitness.begin(), [max_error](float error_) {
                return float_normalized_type(1 - error_ / max_error);
            });

            /// -------------------------------------

            [[maybe_unused]] const auto selected_parents =
                select_parents(generation_variability<float>(m_Population[gen_idx], L2_norm), agent_fitness);

            if (iter == m_Iterations - 1)
            {
                return { m_Population[gen_idx][0], error[0] };
            }

            for (std::size_t i = 0; auto [parent_a, parent_b] : selected_parents)
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
    [[nodiscard]] static auto generation_variability(
        std::array<agent_type, s_Generation_size> const& agents, Distance&& dist_op
    ) -> ga_sm::static_matrix<R, s_Generation_size, s_Generation_size>
    {
        ga_sm::static_matrix<R, s_Generation_size, s_Generation_size> distance_matrix{};
        for (size_t j = 0; j != s_Generation_size - 1; ++j)
        {
            for (size_t i = j + 1; i != s_Generation_size; ++i)
            {
                const auto _distance  = distance<R>(agents[j], agents[i], std::forward<Distance>(dist_op));
                distance_matrix[j, i] = _distance;
                distance_matrix[i, j] = _distance;
            }
        }
        return distance_matrix;
    }

    template <size_t... Is>
    [[nodiscard]] auto generation_factory(std::index_sequence<Is...>) -> std::array<agent_type, sizeof...(Is)>
    {
        return { ((void)Is, make_agent())... };
    }

    [[nodiscard]] auto make_generation() -> generation_container_type
    {
        return generation_factory(std::make_index_sequence<s_Generation_size>());
    }

    [[nodiscard]] auto make_agent() const -> agent_type
    {
        return agent_type(
            brain_type(random::randnormal, 0, 0.5),
            m_Mutation_policy_generator.generate(mutation_policy_generator_type::default_parameters())
        );
    }

    template <typename Distance_Matrix>
        requires(Distance_Matrix::Size == s_Generation_size * s_Generation_size)
    [[nodiscard]] static auto calculate_diversity_scores(Distance_Matrix const& distance_matrix) noexcept
        -> scores_container_type
    {
        const auto cummulative_distance =
            reduce<1, s_Generation_size>(distance_matrix, [](auto a, auto b) { return a + b; });
        const auto                                max_distance = *std::ranges::max_element(cummulative_distance);
        float_normalized_array<s_Generation_size> diversity_scores{};

        for (auto i = 0u; i != s_Generation_size; ++i)
        {
            diversity_scores[i] = float_normalized_type(cummulative_distance[i] / max_distance);
        }
        return diversity_scores;
    }

    // FIXME This is just a test
    // TODO Rework heavily
    [[nodiscard]] static auto calculate_parent_fitness_scores(
        float_normalized_array<s_Generation_size> fitness_scores,
        float_normalized_array<s_Generation_size> diversity_scores
    ) noexcept -> float_normalized_array<s_Generation_size>
    {
        const auto min_fitness   = *std::ranges::min_element(fitness_scores);
        const auto max_fitness   = *std::ranges::max_element(fitness_scores);
        const auto min_diversity = *std::ranges::min_element(diversity_scores);
        const auto max_diversity = *std::ranges::max_element(diversity_scores);

        std::for_each(std::begin(fitness_scores), std::end(fitness_scores), [min_fitness, max_fitness](auto& e) {
            return (e - min_fitness) / (max_fitness - min_fitness);
        });

        std::for_each(
            std::begin(diversity_scores),
            std::end(diversity_scores),
            [min_diversity, max_diversity](auto& e) { return (e - min_diversity) / (max_diversity - min_diversity); }
        );


        float_normalized_array<s_Generation_size> parent_fitness_scores{};

        std::transform(
            fitness_scores.cbegin(),
            fitness_scores.cend(),
            diversity_scores.cbegin(),
            parent_fitness_scores.begin(),
            [](auto a, auto b) -> float_normalized_type {
                return float_normalized_type((float)std::sqrt(a * a + b * b * 0.2) / 1.41422f);
            }
        );

        return parent_fitness_scores;
    }

    template <typename Distance_Matrix>
        requires(Distance_Matrix::Size == s_Generation_size * s_Generation_size)
    [[nodiscard]] auto select_parents(
        Distance_Matrix const& distance_matrix, scores_container_type const& agent_fitness
    ) -> std::array<std::pair<std::size_t, std::size_t>, s_Generation_size / 2>
    {
        std::array<std::size_t, s_Generation_size> indexer{};
        std::iota(indexer.begin(), indexer.end(), 0);
        std::sort(
            indexer.begin(),
            indexer.end(),
            [&agent_fitness](const std::size_t idx_1, const std::size_t idx_2) -> bool {
                return agent_fitness[idx_1] > agent_fitness[idx_2];
            }
        );

        const auto diversity_scores =
            calculate_diversity_scores(distance_matrix); // TODO this should be an input parameter directly?
        const auto parent_fitness_scores = calculate_parent_fitness_scores(agent_fitness, diversity_scores);

        const auto total_fitness = std::accumulate(parent_fitness_scores.cbegin(), parent_fitness_scores.cend(), 0.f);
        auto       cummulative_fitness = parent_fitness_scores;
        for (auto& e : cummulative_fitness)
        {
            e /= total_fitness;
        }
        for (std::size_t i = 1; i != s_Generation_size; ++i) // TODO rework as a function
        {
            cummulative_fitness[i] += cummulative_fitness[i - 1];
        }

        std::array<std::pair<std::size_t, std::size_t>, s_Generation_size / 2> selected_parents{};

        for (auto& e : selected_parents)
        { // TODO Replace with std transform
            const auto a = random::randfloat();
            int        i;
            int        j;
            for (i = 0; i != s_Generation_size; ++i)
            {
                if (cummulative_fitness[i] > a)
                    break;
            }
            do
            {
                const auto b = random::randfloat();
                for (j = 0; j != s_Generation_size; ++j)
                {
                    if (cummulative_fitness[j] > b)
                        break;
                }
            } while (j == i);
            e = { i, j };
        }

        if (random::randfloat() < 0.1)
        {
            std::cout << distance_matrix << '\n';
            std::cout << "Agent fitness: \t";
            for (auto e : agent_fitness)
                std::cout << (float)e << ' ';
            std::cout << '\n';
            std::cout << "Diversity scores: \t";
            for (auto e : diversity_scores)
                std::cout << (float)e << ' ';
            std::cout << '\n';
            std::cout << "Parent fitness: \t";
            for (auto e : parent_fitness_scores)
                std::cout << (float)e << ' ';
            std::cout << '\n';
            for (auto e : cummulative_fitness)
                std::cout << (float)e << ' ';
            std::cout << '\n';
            for (auto e : selected_parents)
            {
                std::cout << e.first << ' ' << e.second << ' ';
                std::cout << " .. ";
            }
            std::cout << '\n';
        }


        return selected_parents;
    }
};

} // namespace tournament_selection

#endif // TOURNAMENT_SELECTION
