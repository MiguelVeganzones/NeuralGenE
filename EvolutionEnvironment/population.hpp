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

template <evolution_environment_traits::agent_concept Agent_Type>
class population
{
public:
    inline static constexpr auto s_Population_generations = 2;
    using generation_idx_type                             = int;
    using agent_type                                      = Agent_Type;
    using generation_container_type = std::vector<agent_type>;
    using population_container_type =
        std::array<generation_container_type, s_Population_generations>;

public:
    template <factory_of<agent_type> Factory>
    population(int generation_size, Factory&& agent_factory) noexcept :
        m_Generation_size{ generation_size },
        m_Population{ make_population(
            m_Generation_size,
            std::forward<Factory>(agent_factory)
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
    auto get_current_generation() const -> generation_container_type const&
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
            << "\nGeneration size: " << m_Generation_size
            << "\n========================================================\n";
        for (int gen_idx = 0; auto&& gen : m_Population)
        {
            std::cout << "\nGen " << gen_idx++ << '\n';
            for (auto&& e : gen)
            {
                e.print();
            }
            std::cout
                << "########################################################\n";
        }
    }

private:
    template <std::size_t... Is, factory_of<agent_type> Agent_Factory_Fn>
        requires(sizeof...(Is) == s_Population_generations)
    [[nodiscard]]
    inline static auto population_factory(
        std::index_sequence<Is...>,
        int                generation_size,
        Agent_Factory_Fn&& agent_factory
    ) -> population_container_type
    {
        return {
            ((void)Is,
             make_generation(
                 generation_size, std::forward<Agent_Factory_Fn>(agent_factory)
             ))...
        };
    }

    template <factory_of<agent_type> Agent_Factory_Fn>
    [[nodiscard]]
    inline static auto make_population(
        int                generation_size,
        Agent_Factory_Fn&& agent_factory
    ) -> population_container_type
    {
        return population_factory(
            std::make_index_sequence<s_Population_generations>(),
            generation_size,
            std::forward<Agent_Factory_Fn>(agent_factory)
        );
    }

    template <factory_of<agent_type> Agent_Factory_Fn>
    [[nodiscard]]
    inline static auto make_generation(
        int                generation_size,
        Agent_Factory_Fn&& agent_factory
    ) -> generation_container_type
    {
        generation_container_type generation;
        for (int i = 0; i != generation_size; ++i)
        {
            generation.emplace_back(std::invoke(agent_factory));
        }
        return generation;
    }

    [[nodiscard]]
    inline auto current_generation_idx() const -> generation_idx_type
    {
        return m_Current_generation_idx;
    }

    [[nodiscard]]
    inline auto next_generation_idx() const -> generation_idx_type
    {
        return static_cast<generation_idx_type>(
            (m_Current_generation_idx + generation_idx_type{ 1 }) %
            s_Population_generations
        );
    }

private:
    generation_idx_type       m_Current_generation_idx = 0;
    int                       m_Generation_size;
    population_container_type m_Population;
};

} // namespace environment_population

#endif // POPULATION
