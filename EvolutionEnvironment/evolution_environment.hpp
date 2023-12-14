#ifndef EVOLUTION_ENVIRONMENT
#define EVOLUTION_ENVIRONMENT

#include "error_handling.hpp"
#include "evolution_environment_traits.hpp"
#include "population.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <type_traits>
#include <utility>
#include <vector>

namespace evolution_env
{

//-----------------------------------------------------------------------------
// ---------------  Evolution environment  ------------------------------------
//-----------------------------------------------------------------------------

template <
    std::uint16_t                                Generation_Size,
    evolution_environment_traits::agent_concept  Agent_Type,
    evolution_environment_traits::system_concept System_Type,
    evolution_environment_traits::reproduction_manager_concept
        Reproduction_Manager_Type>
    requires requires {
        {
            std::declval<System_Type&>().evaluate(
                std::declval<std::array<Agent_Type, Generation_Size>&>()
            )
        };
        // FIXME cannot get this to work
        // } -> std::smae_as<
        // } -> std::convertible_to<
        //     std::ranges::range<typename System_Type::fitness_score_type>>;
    } && std::is_invocable_v<Agent_Type, typename System_Type::stimulus_type>
class evolution_environment

{
public:
    inline static constexpr auto s_Generation_size        = Generation_Size;
    inline static constexpr auto s_Population_generations = 2u;

    using agent_type                = Agent_Type;
    using system_type               = System_Type;
    using reproduction_manager_type = Reproduction_Manager_Type;
    using fitness_score_type        = typename system_type::fitness_score_type;

    using generation_fitness_score_container =
        std::array<float, s_Generation_size>;

    using population_container_type = environment_population::
        population<s_Population_generations, s_Generation_size, agent_type>;
    using result_type = std::pair<agent_type, fitness_score_type>;

public:
    template <environment_population::factory_of<agent_type> Factory>
    evolution_environment(
        Factory&&                        agent_factory,
        System_Type const&               system,
        Reproduction_Manager_Type const& reproduction_manager
    ) noexcept :
        m_Population(std::forward<Factory>(agent_factory)),
        m_System(system),
        m_Reproduction_manager(reproduction_manager)
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
        generation_fitness_score_container generation_fitness{};
        for (auto iter = 0uz; iter != generations; ++iter)
        {
            generation_fitness =
                m_System.evaluate(m_Population.get_current_generation());
            for (auto&& e : generation_fitness)
            {
                std::cout << e << ", ";
            }
            std::cout << std::endl;
            m_Reproduction_manager.yield_next_generation(
                m_Population.get_current_generation(),
                generation_fitness,
                m_Population.get_next_generation_nest()
            );
            m_Population.increment_generation();
        }
        const auto best_fitness_idx = std::distance(
            std::begin(generation_fitness),
            std::ranges::max_element(generation_fitness)
        );
        return { m_Population.get_current_generation()[best_fitness_idx],
                 generation_fitness[best_fitness_idx] };
    }

    auto print_population() const noexcept -> void
    {
        m_Population.print();
    }

private:
    population_container_type m_Population;
    system_type               m_System;
    reproduction_manager_type m_Reproduction_manager;
};

} // namespace evolution_env

#endif // EVOLUTION_ENVIRONMENT
