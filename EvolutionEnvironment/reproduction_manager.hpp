#ifndef REPRODUCTION_MANAGER
#define REPRODUCTION_MANAGER

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

class parent_categories_manager
{
public:
    using restricted_type = generics::containers::restricted<int>;

public:
    parent_categories_manager(int generation_size) noexcept :
        parent_categories_manager(
            generation_size,
            restricted_type(1, (int)(generation_size / 15) + 1, 1),
            2
        )
    {
    }

    parent_categories_manager(
        int             generation_size,
        restricted_type elite_n,
        int             survivors_n
    ) noexcept :
        generation_size_{ generation_size },
        elite_n_{ elite_n },
        survivors_n_{ survivors_n },
        progenitors_n_{
            static_cast<unsigned int>(
                generation_size - elite_n.get_value() - survivors_n
            ) >>
            1 // shift one bit to divide by two and round down
        }
    {
        if (generation_size_ < elite_n_.max() + survivors_n + 2)
        {
            std::cout
                << "Invalid group sizes.  There must be at least space "
                   "for two progenitors at all moments in the generation.\n";
            std::exit(EXIT_FAILURE);
        }
        update_extra_survivor();
    }

    parent_categories_manager(parent_categories_manager const&) noexcept =
        default;
    parent_categories_manager(parent_categories_manager&&) noexcept = default;
    parent_categories_manager&
    operator=(parent_categories_manager const&) noexcept = default;
    parent_categories_manager& operator=(parent_categories_manager&&) noexcept =
        default;

    [[nodiscard]]
    auto elites_count() const noexcept -> int
    {
        return elite_n_.get_value();
    }

    [[nodiscard]]
    auto survivors_count() const noexcept -> int
    {
        return survivors_n_ + extra_survivor();
    }

    [[nodiscard]]
    auto progenitors_count() const noexcept -> int
    {
        return static_cast<int>(progenitors_n_);
    }

    [[nodiscard]]
    auto parents_count() const noexcept -> int
    {
        return elites_count() + survivors_count() + parents_count();
    }

private:
    [[nodiscard]]
    auto extra_survivor() const noexcept -> int
    {
        return extra_survivor_ ? 1 : 0;
    }

    auto update_extra_survivor() noexcept -> void
    {
        extra_survivor_ = generation_size_ >
            (elite_n_.get_value() + survivors_n_ + (int)progenitors_n_ * 2);
    }

    auto increment_elite_count() noexcept -> void
    {
        elite_n_.increment();
        update_extra_survivor();
    }

    auto decrement_elite_count() noexcept -> void
    {
        elite_n_.decrement();
        update_extra_survivor();
    }

private:
    int             generation_size_;
    restricted_type elite_n_;
    int             survivors_n_;
    unsigned int    progenitors_n_;
    bool            extra_survivor_;
};

template <
    int                                         Generation_Size,
    evolution_environment_traits::agent_concept Agent_Type,
    std::floating_point                         Fitness_Score_Type>
class reproduction_manager

{
public:
    inline static constexpr auto s_Generation_size = Generation_Size;

    using diversity_score_type            = float;
    using diversity_scores_container_type = ga_sm::static_matrix<
        diversity_score_type,
        s_Generation_size,
        s_Generation_size>;
    using fitness_score_type        = Fitness_Score_Type;
    using agent_type                = Agent_Type;
    using generation_container_type = std::array<agent_type, s_Generation_size>;
    using generation_fitness_container_type =
        std::array<fitness_score_type, s_Generation_size>;
    using restricted_type = generics::containers::restricted<float>;
    using parent_type =
        std::variant<asexual_reproduction_parent, sexual_reproduction_parents>;

public:
    reproduction_manager() noexcept :
        m_Diversity_weight(0.f, 3.f, 1.0f),
        m_Fitness_weight(1.0f, 3.f, 2.f),
        m_Parent_categories(s_Generation_size)
    {
    }

    reproduction_manager(
        restricted_type           diversity_weight,
        restricted_type           fitness_weight,
        parent_categories_manager parent_categories
    ) noexcept :
        m_Diversity_weight{ diversity_weight },
        m_Fitness_weight{ fitness_weight },
        m_Parent_categories(parent_categories)
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
        auto&& diversity = population_variability<float>(
            current_generation, generics::algorithms::L2_norm
        );
        const auto parents_count =
            update_current_parents(fitness_scores, diversity);
        reproduce_generation(
            parents_count, current_generation, next_generation_nest
        );
    }

private:
    auto update_current_parents(
        generation_fitness_container_type const& fitness_scores,
        diversity_scores_container_type const&   diversity
    ) -> int
    {
        if (random::randfloat() < 0.001)
        {
            std::cout << diversity << std::endl;
        }
        const auto& elite_n_indeces = generics::algorithms::top_n_indeces(
            fitness_scores, m_Parent_categories.elites_count()
        );

        update_best_fitness_score(fitness_scores[elite_n_indeces.back()]);

        generation_fitness_container_type partially_accumulated_fitness{};
        std::partial_sum(
            std::begin(fitness_scores),
            std::end(fitness_scores),
            std::begin(partially_accumulated_fitness)
        );

        // Store actual parents
        int parent_idx = 0;
        for (auto& idx : elite_n_indeces)
        {
            m_Selected_parents[parent_idx++] =
                asexual_reproduction_parent{ idx };
        }
        for (int i = 0; i != m_Parent_categories.survivors_count(); ++i)
        {
            m_Selected_parents[parent_idx++] = asexual_reproduction_parent{
                roulette_select_parent(partially_accumulated_fitness)
            };
        }
        for (int i = 0; i != m_Parent_categories.progenitors_count(); ++i)
        {
            const auto a =
                roulette_select_parent(partially_accumulated_fitness);
            auto b = -1;
            do
            {
                b = roulette_select_parent(partially_accumulated_fitness);
            } while (a == b);

            m_Selected_parents[parent_idx++] =
                sexual_reproduction_parents{ a, b };
        }
        return parent_idx;
    }

    auto reproduce_generation(
        const int                        parents_count,
        generation_container_type const& current_generation,
        generation_container_type&       next_generation_nest
    ) -> void
    {
        // for (int i = 0; i != parents_count; ++i)
        // {
        //     auto&& parent_variant = m_Selected_parents[i];
        //     if (std::holds_alternative<sexual_reproduction_parents>(
        //             parent_variant
        //         ))
        //     {
        //         const auto& parents =
        //             std::get<sexual_reproduction_parents>(parent_variant);
        //         std::cout << parents.a.index << " & " << parents.b.index
        //                   << '\n';
        //     }
        //     else if (std::holds_alternative<asexual_reproduction_parent>(
        //                  parent_variant
        //              ))
        //     {
        //         auto&& parent =
        //             std::get<asexual_reproduction_parent>(parent_variant);
        //         std::cout << parent.p.index << '\n';
        //     }
        // }
        for (int next_gen_idx = 0, parent_idx = 0; parent_idx != parents_count;
             ++parent_idx)
        {
            const auto& parent_variant = m_Selected_parents[parent_idx];
            if (std::holds_alternative<sexual_reproduction_parents>(
                    parent_variant
                ))
            {
                assert(next_gen_idx < s_Generation_size - 1);
                const auto& parents =
                    std::get<sexual_reproduction_parents>(parent_variant);
                to_target_crossover(
                    current_generation[parents.a.index],
                    current_generation[parents.b.index],
                    next_generation_nest[next_gen_idx + 0],
                    next_generation_nest[next_gen_idx + 1]
                );
                next_gen_idx += 2;
            }
            else if (std::holds_alternative<asexual_reproduction_parent>(
                         parent_variant
                     ))
            {
                assert(next_gen_idx < s_Generation_size);
                auto&& parent =
                    std::get<asexual_reproduction_parent>(parent_variant);
                next_generation_nest[next_gen_idx++] =
                    current_generation[parent.p.index];
            }
            else
            {
                assert_unreachable();
            }
        }
        // todo fix
        for (int i = m_Parent_categories.elites_count(); i != parents_count;
             ++i)
        {
            next_generation_nest[i].mutate();
        }
    }

    auto update_best_fitness_score(fitness_score_type top_score) noexcept
        -> void
    {
        if (top_score > m_Best_score)
        {
            m_Best_score = top_score;
            /*
            TODO
            */
        }
    }

    auto roulette_select_parent(
        generation_fitness_container_type const& partially_accumulated_fitness
    ) const -> int
    {
        // TODO make fitness allways positive and better the greater
        auto r = random::randfloat() * partially_accumulated_fitness.back();
        auto rand_idx = std::distance(
            std::begin(partially_accumulated_fitness),
            std::lower_bound(
                std::begin(partially_accumulated_fitness),
                std::end(partially_accumulated_fitness),
                r
            )
        );

        return static_cast<int>(rand_idx);
    }

private:
    std::array<parent_type, s_Generation_size> m_Selected_parents{};
    fitness_score_type                         m_Best_score =
        std::numeric_limits<fitness_score_type>::min();
    diversity_score_type m_Reference_diversity_score =
        std::numeric_limits<diversity_score_type>::min();
    restricted_type           m_Diversity_weight;
    restricted_type           m_Fitness_weight;
    parent_categories_manager m_Parent_categories;
};

} // namespace reproduction_mngr


#endif // REPRODUCTION_MANAGER