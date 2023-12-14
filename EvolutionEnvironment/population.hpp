#ifndef POPULATION
#define POPULATION

#include <array>
#include <concepts>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace environment_population
{

template <typename Factory, typename Agent>
concept factory_of = requires(Factory f) {
    {
        f()
    } -> std::same_as<Agent>;
};

//-----------------------------------------------------------------------------
// ---------------  Population  -----------------------------------------------
//-----------------------------------------------------------------------------

template <
    std::uint8_t                                Generation_Count,
    std::uint16_t                               Generation_Size,
    evolution_environment_traits::agent_concept Agent_Type>
class population
{
public:
    inline static constexpr auto s_Generation_size        = Generation_Size;
    inline static constexpr auto s_Population_generations = Generation_Count;
    using agent_type                                      = Agent_Type;
    using generation_container_type = std::array<agent_type, s_Generation_size>;
    using population_container_type =
        std::array<generation_container_type, s_Population_generations>;
    using generation_idx_type = decltype(Generation_Count);


public:
    population() noexcept
        requires std::is_default_constructible_v<agent_type>
        :
        m_Population{ make_population([]() -> agent_type {
            return agent_type{};
        }) }

    {
    }

    template <factory_of<agent_type> Factory>
    population(Factory&& agent_factory) noexcept :
        m_Population{ make_population(std::forward<Factory>(agent_factory)) }

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

    [[nodiscard]]
    auto get_next_generation_nest() -> generation_container_type&
    {
        return m_Population[next_generation_idx()];
    }

    auto increment_generation() -> void
    {
        m_Current_generation_idx = next_generation_idx();
    }

    auto print() const noexcept -> void
    {
        std::cout
            << "========================================================\n"
            << "Population generations: " << s_Population_generations
            << "\nGeneration size:" << s_Generation_size
            << "\n========================================================\n";
        for (int gen_idx = 0; auto&& gen : m_Population)
        {
            std::cout << "Gen " << gen_idx++ << '\n';
            for (auto&& e : gen)
            {
                e.print();
            }
            std::cout
                << "########################################################";
        }
    }

private:
    template <std::size_t... Is, factory_of<agent_type> Agent_Factory_Fn>
        requires(sizeof...(Is) == s_Population_generations)
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

    template <factory_of<agent_type> Agent_Factory_Fn>
    [[nodiscard]]
    inline static auto make_population(Agent_Factory_Fn&& agent_factory)
        -> population_container_type
    {
        return population_factory(
            std::make_index_sequence<s_Population_generations>(),
            std::forward<Agent_Factory_Fn>(agent_factory)
        );
    }

    template <std::size_t... Is, factory_of<agent_type> Agent_Factory_Fn>
        requires(sizeof...(Is) == s_Generation_size)
    [[nodiscard]]
    inline static auto generation_factory(
        std::index_sequence<Is...>,
        Agent_Factory_Fn&& agent_factory
    ) -> generation_container_type
    {
        return { ((void)Is, std::invoke(agent_factory))... };
    }

    template <factory_of<agent_type> Agent_Factory_Fn>
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
        return static_cast<generation_idx_type>(
            (m_Current_generation_idx + generation_idx_type{ 1 }) %
            s_Population_generations
        );
    }

private:
    generation_idx_type       m_Current_generation_idx = 0;
    population_container_type m_Population;
};

} // namespace environment_population

#endif // POPULATION