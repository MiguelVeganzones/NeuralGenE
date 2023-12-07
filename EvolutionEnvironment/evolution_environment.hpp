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
    T::Population_Size;
    {
        t.yiled_next_generation(
            ga_sm::static_matrix<float, T::Population_Size, T::Population_Size>,
            std::array<float, T::Population_Size>,
            std::array<typename T::agent_type, T::Population_Size> const&
                current_generation,
            std::array<typename T::agent_typr, T::Population_Size>&
                next_generation
        )
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
    inline constexpr auto s_Generation_size        = Genration_Size;
    inline constexpr auto s_Population_generations = Genration_Count;
    using generation_container_type = std::array<agent_type, s_Generation_size>;
    using populaiton_container_type =
        std::array<generation_container_type, s_Populaiton_generations>;
    using generation_idx_type = decltype(Generation_Count);
    using agent_type          = Agent_Type;

public:
    auto get_current_generation() -> generation_container_type const&
    {
        return m_Population[current_generation_idx()];
    }

    auto get_nex_generation_nest() -> generation_container_type&
    {
        return m_Population[next_generation_idx()];
    }

    auto increment_generation() -> void
    {
        m_Current_generation_idx = (m_Current_generation_idx + 1) %=
            s_Population_generations;
    }

private:
    inline [[nodiscard]]
    auto current_generation_idx() -> generation_idx_type
    {
        return m_Current_generation_idx;
    }

    inline [[nodiscard]]
    auto next_generation_idx() -> generation_idx_type
    {
        return (m_Current_generation_idx + 1) % s_Population_geneation_size;
    }

private:
    std::uint8_t              m_Current_generation_index = 0;
    populaiton_container_type m_Population;
}

//-----------------------------------------------------------------------------
// ---------------  Evolution environment  ------------------------------------
//-----------------------------------------------------------------------------

template <
    std::uint16_t                Generation_Size,
    agent_concept                Agent_Type,
    system_concept               System_Type,
    reproduction_manager_concept Reproduction_Manager_Type>
    requires requires {
        {
            System_Type::evaluate(std::range<Agent_Type> const&)
        } -> std::same_as <
            std::range<typename System_Type::fitness_score_type>;
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

    using generation_fitness_score_container =
        std::array<float, s_Genreation_size>;
    using population_container_type = population;
    using system_type               = System_Type;

private:
    population_container_type m_Population;
};

} // namespace evolution_environemnt

#endif // EVOLUTION_ENVIRONMENT
