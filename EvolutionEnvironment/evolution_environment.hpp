#ifndef EVOLUTION_ENVIRONMENT
#define EVOLUTION_ENVIRONMENT

#include "error_handling.hpp"
#include "static_matrix.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <type_traits>
#include <vector>

namespace evolution_environemnt
{

template <typename T>
concept agent_concept = requires(T t) {
    typename T::brain_value_type;
    typename T::agent_input_type;
    typename T::agent_output_type;
    {
        t.mutate()
    } -> std::same_as<void>;
    {
        to_target_crossover(t, t, t, t)
    } -> std::same_as<void>;
    {
        t(std::declval<typename T::agent_input_type&>)
    } -> std::same_as<typename T::agent_output_type>;
    {
        population_variability<
            float>(std::array<T, std::declval<std::uint16_t>()>)
    } -> ga_sm::static_matrix<
        float,
        std::declval<std::uint16_t>(),
        std::declval<std::uint16_t>()>;
};

template <typename T>
concept system_concept = requires(T t) {
    typename T::system_stimulus_type;
    typename T::agent_response_type;
    typename T::fitness_score_type;
    // FIXME in the future
    // require system to be able to evaluate a range of agents. Cannot do it
    // with current concepts without using an archetype agent
};

template <typename T>
concept reproduction_manager_concept = requires(T t) {
    typename T::agent_type;
    T::s_Population_Size;
    {
        t.produce_next_generation(std::array<typename T::agent_type, T::s_Population_Size> const&, std::array<float, T::s_Population_Size>, std::array<typename T::agent_typr, T::s_Population_Size>&)
    } -> void;
};

//-----------------------------------------------------------------------------
// ---------------  Population  -----------------------------------------------
//-----------------------------------------------------------------------------

template <
    std::uint8_t  Generation_Count,
    std::uint16_t Generation_Size,
    agent_concept Agent_Type>
class population
{
public:
    inline static constexpr auto s_Generation_size        = Genration_Size;
    inline static constexpr auto s_Population_generations = Genration_Count;
    using generation_container_type = std::array<agent_type, s_Generation_size>;
    using populaiton_container_type =
        std::array<generation_container_type, s_Populaiton_generations>;
    using generation_idx_type = decltype(Generation_Count);
    using agent_type          = Agent_Type;

    inline static constexpr auto agent_factory_concept = requires(auto t) {
        {
            t.make_agent()
        } -> std::same_as<agent_type>;
    };

public:
    population() noexcept
        requires std::is_default_constructible_v<agent_type>
        :
        m_Population{ make_population([]() -> agent_type {
            return Agent_type{};
        }) }

    {
    }

    population(agent_factory_concept auto&& agent_factory) noexcept :
        m_Population{ make_population(
            std::forward<decltype(agent_factory)>(agent_factory)
        ) }

    {
    }

    population(population const&) noexcept            = default;
    population(population&&) noexcept                 = default;
    population& operator=(population const&) noexcept = default;
    population& operator=(population&&) noexcept      = default;
    ~population() noexcept                            = default;

public:
    [[nodiscard]]
    auto get_current_generation() -> generation_container_type const&
    {
        return m_Population[current_generation_idx()];
    }

    [[nodsicard]]
    auto get_next_generation_nest() -> generation_container_type&
    {
        return m_Population[next_generation_idx()];
    }

    auto increment_generation() -> void
    {
        m_Current_generation_idx = (m_Current_generation_idx + 1) %=
            s_Population_generations;
    }

private:
    template <std::size_t... Is, agent_factory_concept Agent_Factory_Fn>
        requires(sizeof...(Is) == s_Population_size)
    [[nodiscard]]
    inline static auto population_factory(
        std::index_sequence<Is...>,
        Agent_Factory_Fn&& agent_factory
    ) -> population_container_type
    {
        return {
            ((void)Is,
             make_generation(std::forward<Agent_Factory_Fn>(agent_factory)))...
        };
    }

    [[nodiscard]]
    template <agent_factory_concept Agent_Factory_Fn>
    inline static auto make_population(Agent_Factory_Fn&& agent_factory)
        -> population_container_type
    {
        return {
            ((void)Is,
             make_generation(std::forward<Agent_Factory_Fn>(agent_factory)))...
        };
    }

    template <std::size_t... Is, agent_factory_concept Agent_Factory_Fn>
        requires(sizeof...(Is) == s_Generation_size)
    [[nodiscard]]
    inline static auto generation_factory(
        std::index_sequence<Is...>,
        Agent_Factory_Fn&& agent_factory
    ) -> generation_container_type
    {
        return { ((void)Is, std::invoke(agent_factory))... };
    }

    template <agent_factory_concept Agent_Factory_Fn>
    [[nodiscard]]
    inline static auto make_generation(Agent_Factory_Fn&& agent_factory)
        -> generation_container_type
    {
        return generation_factory(
            std::make_index_sequence<s_Generation_size>(),
            std::forward<Agent_Factory_Fn>(agent_factory)
        );
    }

    [[nodiscard]]
    inline auto current_generation_idx() -> generation_idx_type
    {
        return m_Current_generation_idx;
    }

    [[nodiscard]]
    inline auto next_generation_idx() -> generation_idx_type
    {
        return (m_Current_generation_idx + 1) % s_Population_geneation_size;
    }

private:
    std::uint8_t              m_Current_generation_index = 0;
    populaiton_container_type m_Population;
}

//-----------------------------------------------------------------------------
// ---------------  Evolution environment
// ------------------------------------
//-----------------------------------------------------------------------------

template <
    std::uint16_t                Generation_Size,
    agent_concept                Agent_Type,
    system_concept               System_Type,
    reproduction_manager_concept Reproduction_Manager_Type>
    requires requires {
        {
            System_Type::evaluate(std::range<Agent_Type> const&)
        } -> std::same_as<std::range<typename System_Type::fitness_score_type>>;
    } &&
    std::is_invocable_r_v<
                 typename System_Type::agent_response_type,
                 std::declval<Agent_Type&>(),
                 typename System_Type::system_stimulus_type>
class evolution_environment

{
public:
    inline constexpr auto s_Generation_size        = Generation_Size;
    inline constexpr auto s_Population_generations = 2u;

    using agent_type                = Agnet_Type;
    using system_type               = System_Type;
    using reproduction_manager_type = Reproduction_Manager_Type;
    using fitness_score_type        = typename system_type::fitness_score_type;

    using generation_fitness_score_container =
        std::array<float, s_Genreation_size>;
    using population_container_type = population;
    using system_type               = System_Type;
    using result_type               = std::pair<agent_type, fitness_score_type>;

public:
    evolution_environment() noexcept :
        m_Population()
    {
    }

    evolution_environment(population::agent_factory_concept auto&& agent_factory
    ) noexcept :
        m_population(std::forward<decltype(agent_factory)>(agent_factory))
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
        for (auto iter = 0; iter != generations; ++iter)
        {
            const auto generation_fitness =
                system_type::evaluate(m_Population.get_current_generation());
            m_Reproduction_manager.produce_next_generation(
                m_Population.get_current_generation(),
                generation_fitness,
                m_Population.get_next_generation_nest()
            );
            if (iter == generations - 1)
            {
                const auto best_fitness_idx = std::distance(
                    std::begin(generation_fitness),
                    std::ranges::max_element(generation_fitness)
                );
                return { m_Population.get_current_generation(
                         )[best_fitness_idx],
                         generation_fitness[best_fitness_idx] };
            }
        }
        assert_unreachable();
    }

private:
    population_container_type m_Population;
    reproduction_manager_type m_Reproduction_manager;
};

} // namespace evolution_environemnt

#endif // EVOLUTION_ENVIRONMENT
