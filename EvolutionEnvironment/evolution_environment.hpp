#ifndef EVOLUTION_ENVIRONMENT
#define EVOLUTION_ENVIRONMENT

#include "error_handling.hpp"
#include "evolution_environment_traits.hpp"
#include "population.hpp"
#include "reproduction_manager.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <execution>
#include <type_traits>
#include <utility>
#include <vector>

namespace evolution_env
{

template <
    evolution_environment_traits::agent_concept Agent_Type,
    evolution_environment_traits::reproduction_manager_concept
        Reproduction_Manager_Type>
class deme
{
public:
    using agent_type = Agent_Type;
    using population_container_type =
        environment_population::population<agent_type>;
    using generation_container_type =
        typename population_container_type::generation_container_type;
    using reproduction_manager_type = Reproduction_Manager_Type;
    template <std::floating_point T>
    using generation_fitness_score_container = std::vector<T>;

public:
    template <environment_population::factory_of<agent_type> Agent_Factory>
    deme(
        int                                  deme_size,
        Agent_Factory                        agent_factory,
        reproduction_mngr::parent_categories parent_categories
    ) noexcept :
        m_Population(deme_size, std::forward<Agent_Factory>(agent_factory)),
        m_Reproduction_manager(parent_categories)
    {
    }

    template <std::floating_point T>
    inline auto step(
        generation_fitness_score_container<T> const& generation_fitness
    ) -> void
    {
        m_Reproduction_manager.yield_next_generation(
            m_Population.get_current_generation(),
            generation_fitness,
            m_Population.get_next_generation_nest()
        );
        m_Population.increment_generation();
    }

    [[nodiscard]]
    auto get_current_generation() const noexcept -> generation_container_type
    {
        return m_Population.get_current_generation();
    }

    auto print() const noexcept -> void
    {
        m_Population.print();
    }

private:
    population_container_type m_Population;
    reproduction_manager_type m_Reproduction_manager;
};

//-----------------------------------------------------------------------------
// ---------------  Evolution environment
// ------------------------------------
//-----------------------------------------------------------------------------

template <
    evolution_environment_traits::agent_concept  Agent_Type,
    evolution_environment_traits::system_concept System_Type,
    evolution_environment_traits::reproduction_manager_concept
        Reproduction_Manager_Type>
    requires requires {
        {
            std::declval<System_Type&>().evaluate(
                std::declval<std::vector<Agent_Type>&>()
            )
        };
        // FIXME cannot get this to work
        // } -> std::smae_as<
        // } -> std::convertible_to<
        //     std::ranges::range<typename
        //     System_Type::fitness_score_type>>;
    } && std::is_invocable_v<Agent_Type, typename System_Type::stimulus_type>
class evolution_environment

{
public:
    inline static constexpr auto s_Population_generations = 2u;

    using agent_type                = Agent_Type;
    using system_type               = System_Type;
    using reproduction_manager_type = Reproduction_Manager_Type;
    using fitness_score_type        = typename system_type::fitness_score_type;

    using generation_fitness_score_container = std::vector<fitness_score_type>;

    using population_container_type =
        environment_population::population<agent_type>;
    using result_type = std::pair<agent_type, fitness_score_type>;
    using deme_type   = deme<agent_type, reproduction_manager_type>;

public:
    template <environment_population::factory_of<agent_type> Factory>
    evolution_environment(
        int                                  deme_count,
        int                                  deme_size,
        Factory&&                            agent_factory,
        System_Type const&                   system,
        reproduction_mngr::parent_categories parent_categories
    ) noexcept :
        m_Deme_count{ deme_count },
        m_Deme_size{ deme_size },
        m_System(system),
        m_Demes(
            m_Deme_count,
            deme_type(
                deme_size,
                std::forward<Factory>(agent_factory),
                parent_categories
            )
        )
    {
    }

    evolution_environment(evolution_environment const&) noexcept = default;
    evolution_environment(evolution_environment&&) noexcept      = default;
    evolution_environment& operator=(evolution_environment const&) noexcept =
        default;
    evolution_environment& operator=(evolution_environment&&) noexcept =
        default;
    ~evolution_environment() noexcept = default;

    [[nodiscard]]
    inline auto train(std::size_t generations) -> result_type
    {
        std::for_each(
            std::execution::par_unseq,
            std::begin(m_Demes),
            std::end(m_Demes),
            [generations, this](auto& deme) {
                std::cout << "Here\n";
                auto generation_fitness =
                    m_System.evaluate(deme.get_current_generation());
                for (auto iter = 0uz; iter != generations; ++iter)
                {
                    deme.step(generation_fitness);
                    generation_fitness =
                        m_System.evaluate(deme.get_current_generation());
                }
                /*const auto best_fitness_idx = std::distance(
                    std::cbegin(generation_fitness),
                    std::ranges::max_element(generation_fitness)
                );*/
            }
        );
        return { m_Demes[0].get_current_generation()[0], 1.0f };
    }

    auto print_population() const noexcept -> void
    {
        for (int idx = 0; const auto& deme : m_Demes)
        {
            std::cout << "\nDeme #" << idx++
                      << "\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
                         "@@@@@@@@@@@@@@@@@\n";
            deme.print();
        }
    }

private:
    int                    m_Deme_count;
    int                    m_Deme_size;
    system_type            m_System;
    std::vector<deme_type> m_Demes;
};

} // namespace evolution_env

#endif // EVOLUTION_ENVIRONMENT
