#ifndef REPRODUCTION_MANAGER
#define REPRODUCTION_MANAGER

#include "Random.hpp"
#include "evolution_environment_traits.hpp"
#include "generics.hpp"
#include "static_matrix.hpp"
#include <array>
#include <concepts>
#include <cstdlib>
#include <limits>
#include <ranges>
#include <type_traits>
#include <variant>

namespace reproduction_mngr
{

template <typename T>
concept mutation_policy_concept = requires(T t) {
    typename T::value_type;
    typename T::parameters_type;

    {
        std::declval<T&>()(std::declval<typename T::value_type>())
    } -> std::same_as<typename T::value_type>;
};

struct parent
{
    int index = -1;
};

struct asexual_reproduction_parent
{
    parent p;
};

struct sexual_reproduction_parents
{
    parent a;
    parent b;
};

class parent_categories
{
public:
    parent_categories(
        int gen_size,
        int elites_count,
        int survivor_count
    ) noexcept :
        elites_n{ elites_count },
        progenitors_n{
            (int)((unsigned int)(gen_size - elites_n - survivor_count) >> 1)
        },
        survivors_n(gen_size - elites_n - 2 * progenitors_n)
    {
    }

    [[nodiscard]]
    auto elites_count() const noexcept -> int
    {
        return elites_n;
    }

    [[nodiscard]]
    auto progenitors_count() const noexcept -> int
    {
        return progenitors_n;
    }

    [[nodiscard]]
    auto survivors_count() const noexcept -> int
    {
        return survivors_n;
    }

    [[nodiscard]]
    auto generation_size() const noexcept -> int
    {
        return elites_count() + progenitors_count() * 2 + survivors_count();
    }

private:
    int elites_n;
    int progenitors_n;
    int survivors_n;
};

template <
    evolution_environment_traits::agent_concept Agent_Type,
    std::floating_point                         Fitness_Score_Type,
    mutation_policy_concept                     Mutation_Policy>
class reproduction_manager

{
public:
    using diversity_score_type = float;
    using diversity_scores_container_type =
        std::vector<std::vector<diversity_score_type>>;
    using fitness_score_type        = Fitness_Score_Type;
    using agent_type                = Agent_Type;
    using mutation_policy_type      = Mutation_Policy;
    using generation_container_type = std::vector<agent_type>;
    template <typename T>
    using container_type                    = std::vector<T>;
    using generation_fitness_container_type = std::vector<fitness_score_type>;
    using restricted_type = generics::containers::restricted<float>;
    using parent_type =
        std::variant<asexual_reproduction_parent, sexual_reproduction_parents>;

public:
    reproduction_manager(parent_categories parent_categories) noexcept :
        m_Generation_size{ parent_categories.generation_size() },
        m_Parent_categories(parent_categories),
        m_Asexual_reproduction_parents(
            m_Parent_categories.elites_count() +
            m_Parent_categories.survivors_count()
        ),
        m_Sexual_reproduction_parents(m_Parent_categories.progenitors_count()),
        m_Diversity_scores(m_Generation_size),
        m_Fitness_scores(m_Generation_size),
        m_Mutation_policy(m_Base_probability)
    {
    }

    reproduction_manager(reproduction_manager const&) noexcept = default;
    reproduction_manager(reproduction_manager&&) noexcept      = default;
    reproduction_manager& operator=(reproduction_manager const&) noexcept =
        default;
    reproduction_manager& operator=(reproduction_manager&&) noexcept = default;

    // TODO write alternative for no generatio variability avilable
    auto yield_next_generation(
        generation_container_type const&         current_generation,
        generation_fitness_container_type const& fitness_scores,
        generation_container_type&               next_generation_nest
    ) -> void
        requires requires {
            {
                population_variability<diversity_score_type>(
                    std::declval<generation_container_type&>(),
                    generics::algorithms::L2_norm
                )
            } -> std::same_as<diversity_scores_container_type>;
        }
    {
        const auto& diversity = population_variability<diversity_score_type>(
            current_generation, generics::algorithms::L2_norm
        );
        update_current_parents(fitness_scores, diversity);
        reproduce_generation(current_generation, next_generation_nest);
    }

private:
    auto update_current_parents(
        generation_fitness_container_type const& fitness_scores,
        diversity_scores_container_type const&   diversity
    ) -> void
    {
        if (m_Random_engine.randfloat() < 0.01f)
        {
            std::cout << "[\n";
            for (auto& v : diversity)
            {
                for (auto& e : v)
                {
                    std::cout << e << ", ";
                }
                std::cout << '\n';
            }
            std::cout << "]\n";
        }
        //
        // reset diversity scores
        reset_internal_state();

        update_normalized_fitness_scores(fitness_scores);

        if (m_Parent_categories.elites_count())
        {
            const auto& elite_n_indeces = generics::algorithms::top_n_indeces(
                fitness_scores, m_Parent_categories.elites_count()
            );
            update_best_fitness_score(fitness_scores[elite_n_indeces.front()]);
            for (auto& idx : elite_n_indeces)
            {
                add_parent(asexual_reproduction_parent{ idx }, diversity);
            }
        }
        else
        {
            update_best_fitness_score(std::ranges::max(fitness_scores));
        }

        // std::partial_sum(
        //     std::begin(fitness_scores),
        //     std::end(fitness_scores),
        //     std::begin(partially_accumulated_fitness)
        // );


        // funcion add parent/s que actualice current_parents, diversity scores
        // y modified fitness(?) y discretice fitness si tal


        for (int i = 0; i != m_Parent_categories.survivors_count(); ++i)
        {
            auto idx = roulette_select_parent();
            add_parent(asexual_reproduction_parent{ idx }, diversity);
        }
        for (int i = 0; i != m_Parent_categories.progenitors_count(); ++i)
        {
            const auto a = roulette_select_parent();
            auto       b = -1;
            do
            {
                b = roulette_select_parent();
            } while (a == b);

            add_parent(sexual_reproduction_parents{ a, b }, diversity);
        }
    }

    auto reset_internal_state() noexcept
    {
        std::ranges::fill(m_Diversity_scores, 0);
        m_Asexual_parents_idx = 0;
        m_Sexual_parents_idx  = 0;
    }

    auto add_parent(
        asexual_reproduction_parent            parent,
        diversity_scores_container_type const& diversity
    ) noexcept -> void
    {
        m_Asexual_reproduction_parents[m_Asexual_parents_idx++] = parent;
        for (auto j = 0; j != m_Generation_size; ++j)
        {
            m_Diversity_scores[j] += diversity[j][parent.p.index];
        }
        // std::cout << "Diversity_scores " << parent.p.index << "\t";
        // for (auto& e : m_Diversity_scores)
        // {
        //     std::cout << e << " ";
        // }
        // std::cout << '\n';
    }

    auto add_parent(
        sexual_reproduction_parents            parents,
        diversity_scores_container_type const& diversity
    ) noexcept -> void
    {
        m_Sexual_reproduction_parents[m_Sexual_parents_idx++] = parents;
        for (auto j = 0; j != m_Generation_size; ++j)
        {
            m_Diversity_scores[j] +=
                diversity[j][parents.a.index] + diversity[j][parents.b.index];
        }
        // std::cout << "Diversity_scores " << parents.a.index << " "
        //           << parents.b.index << "\t";
        // for (auto& e : m_Diversity_scores)
        // {
        //     std::cout << e << " ";
        // }
        // std::cout << '\n';
    }

    auto update_normalized_fitness_scores(
        generation_fitness_container_type const& fitness_scores
    ) -> void
    {
        std::vector<int> indeces(m_Generation_size);
        std::ranges::iota(indeces, 0);
        std::ranges::sort(indeces, [&fitness_scores](int a, int b) {
            return fitness_scores[a] > fitness_scores[b];
        });
        for (int i = 0; i != m_Generation_size; ++i)
        {
            m_Fitness_scores[indeces[i]] = (float)std::pow(0.97f, i);
        }
        // std::cout << "Fitness\t\t";
        // for (auto& e : fitness_scores)
        // {
        //     std::cout << e << " ";
        // }
        // std::cout << '\n';
        // std::cout << "indeces\t\t";
        // for (auto& e : indeces)
        // {
        //     std::cout << e << " ";
        // }
        // std::cout << '\n';
        // std::cout << "Normalized fitness ";
        // for (auto& e : m_Fitness_scores)
        // {
        //     std::cout << e << " ";
        // }
        // std::cout << '\n';
    }

    auto reproduce_generation(
        generation_container_type const& current_generation,
        generation_container_type&       next_generation_nest

    ) -> void
    {
        int next_gen_idx = 0;
        for (auto parent : m_Asexual_reproduction_parents)
        {
            next_generation_nest[next_gen_idx++] =
                current_generation[parent.p.index];
        }
        for (auto parents : m_Sexual_reproduction_parents)
        {
            to_target_crossover(
                current_generation[parents.a.index],
                current_generation[parents.b.index],
                next_generation_nest[next_gen_idx + 0],
                next_generation_nest[next_gen_idx + 1]
            );
            next_gen_idx += 2;
        }
        // // TODO fix
        // for (int i = m_Parent_categories.elites_count(); i !=
        // s_Generation_size;
        //      ++i)
        for (int i = m_Parent_categories.elites_count(); i != m_Generation_size;
             ++i)
        {
            next_generation_nest[i].mutate(m_Mutation_policy);
        }
    }

    auto update_best_fitness_score(fitness_score_type top_score) noexcept
        -> void
    {

        static constexpr
            typename mutation_policy_type::value_type delta{ 0.000005f };
        static constexpr
            typename mutation_policy_type::value_type min{ 0.0005f };
        if (top_score > m_Best_score * 1.001f)
        {
            m_Base_probability = min;
            m_Best_score       = top_score;
        }
        else
        {
            if (m_Random_engine.randfloat() < 0.0005)
            {
                m_Base_probability = min;
            }
            else
            {
                m_Base_probability += delta;
            }
        }
        m_Mutation_policy.set_base_probability(m_Base_probability);
        // if (top_score < m_Best_score)
        // {
        //     std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n";
        // }
        // std::cout << m_Base_probability << '\n';
        // std::cout << top_score << '\n';
    }

    auto roulette_select_parent() const -> int
    {
        const auto max_diversity = std::ranges::max(m_Diversity_scores);

        container_type<diversity_score_type> m_Modified_scores(m_Generation_size
        );

        m_Modified_scores[0] = generics::algorithms::L2_norm(
            m_Fitness_scores[0], m_Diversity_scores[0] / max_diversity
        );
        for (int i = 1; i != m_Generation_size; ++i)
        {
            m_Modified_scores[i] =
                generics::algorithms::L2_norm(
                    m_Fitness_scores[i], m_Diversity_scores[i] / max_diversity
                ) +
                m_Modified_scores[i - 1];
        }
        // std::cout << "diversity scores:\t";
        // for (auto& e : m_Diversity_scores)
        // {
        //     std::cout << e << ' ';
        // }
        // std::cout << '\n';
        // std::cout << "Modified fitness:\t";
        // for (auto& e : m_Modified_scores)
        // {
        //     std::cout << e << ' ';
        // }
        // std::cout << '\n';
        auto rand_idx = std::distance(
            std::begin(m_Modified_scores),
            std::lower_bound(
                std::begin(m_Modified_scores),
                std::end(m_Modified_scores),
                m_Random_engine.randfloat() * m_Modified_scores.back()
            )
        );
        return static_cast<int>(rand_idx);
    }

private:
    int                m_Generation_size;
    fitness_score_type m_Best_score =
        std::numeric_limits<fitness_score_type>::min();
    parent_categories                         m_Parent_categories;
    std::vector<asexual_reproduction_parent>  m_Asexual_reproduction_parents;
    std::vector<sexual_reproduction_parents>  m_Sexual_reproduction_parents;
    container_type<diversity_score_type>      m_Diversity_scores;
    container_type<diversity_score_type>      m_Fitness_scores;
    int                                       m_Asexual_parents_idx = 0;
    int                                       m_Sexual_parents_idx  = 0;
    typename mutation_policy_type::value_type m_Base_probability{ 0.0001f };
    mutation_policy_type                      m_Mutation_policy;
    mutable random_::random                   m_Random_engine{};
};

} // namespace reproduction_mngr


#endif // REPRODUCTION_MANAGER