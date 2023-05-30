#ifndef EVOLUTION_AGENT
#define EVOLUTION_AGENT

#include "Random.h"
#include "data_processor.hpp"
#include "neural_model.hpp"
#include "score_functions.hpp"
#include <iostream>

namespace evolution_agent
{

template <ga_neural_model::brain_type Brain>
class agent
{
public:
    using brain_type             = Brain;
    using id_type                = std::size_t;
    using generation_type        = std::size_t;
    using parents_container_type = std::vector<id_type>; // TODO: Optimize(?)
    using agent_output_type      = brain_type::brain_output_type;
    using brain_value_type       = brain_type::value_type;

private:
    inline static std::atomic<id_type> s_ID = 1;

    id_type                m_ID = s_ID++;
    generation_type        m_Generation{};
    parents_container_type m_Parents{};
    brain_type             m_Brain{};


private:
    agent(const brain_type& brain, const parents_container_type& parents, generation_type generation) :
        m_Generation{ generation },
        m_Parents{ parents },
        m_Brain(brain)
    {
    }

    agent(brain_type&& brain, const parents_container_type& parents, generation_type generation) :
        m_Generation{ generation },
        m_Parents{ parents },
        m_Brain(std::move(brain))
    {
    }

public:
    agent(brain_type&& brain) noexcept :
        m_Brain(std::move(brain))
    {
    }

    agent() noexcept                        = default;
    agent(const agent&) noexcept            = default;
    agent(agent&&) noexcept                 = default;
    agent& operator=(const agent&) noexcept = default;
    agent& operator=(agent&&) noexcept      = default;
    ~agent() noexcept                       = default;

    //--------------------------------------------------------------------------------------//
    // Agent reproduction
    //--------------------------------------------------------------------------------------//

    /// @brief Asexual reproduction by cloning. Generates an identical agent member of the next generation.
    /// @return New Agent member of the next generation to the parent.
    [[nodiscard]] auto clone() const -> agent
    {
        return agent(m_Brain, { m_ID }, m_Generation + 1);
    }

    template <typename Fn>
        requires std::is_invocable_r_v<brain_value_type, Fn, brain_value_type>
    void mutate(Fn&& fn)
    {
        m_Brain.mutate(fn);
    }

    [[nodiscard]] static auto crossover(const agent& agent_a, const agent& agent_b) -> std::pair<agent, agent>
    {
        // TODO: Maybe try to_target_net_x_crossover
        auto       brains     = brain_crossover(agent_a.m_Brain, agent_b.m_Brain);
        const auto parents    = parents_container_type{ agent_a.ID(), agent_b.ID() };
        const auto generation = get_offsprings_generation(agent_a.generation(), agent_b.generation());
        return { agent(std::move(brains.first), parents, generation),
                 agent(std::move(brains.second), parents, generation) };
    }

    static auto in_place_crossover(agent& agent_a, agent& agent_b) -> void
    {
        in_place_brain_crossover(agent_a.m_Brain, agent_b.m_Brain);
    }

    static auto to_target_crossover(const agent& parent_a, const agent& parent_b, agent& child_a, agent& child_b)
        -> void
    {
        to_target_brain_crossover(parent_a.m_Brain, parent_b.m_Brain, child_a.m_Brain, child_b.m_Brain);
    }

    //--------------------------------------------------------------------------------------//
    // Agent interface with its environment
    //--------------------------------------------------------------------------------------//

    template <typename Agent_Input_Type>
        requires requires { m_Brain(std::declval<Agent_Input_Type>()); }
    [[nodiscard]] auto operator()(const Agent_Input_Type& input) const -> agent_output_type
    {
        return m_Brain(input);
    }

    //--------------------------------------------------------------------------------------//
    // Utility
    //--------------------------------------------------------------------------------------//


    [[nodiscard]] auto ID() const -> id_type
    {
        return m_ID;
    }

    [[nodiscard]] auto generation() const -> generation_type
    {
        return m_Generation;
    }

    // [[nodiscard]] static auto get_offsprings_generation(generation_type gen_a, generation_type gen_b) ->
    // generation_type
    // {
    //     return static_cast<generation_type>(static_cast<float>(gen_a + gen_b) / 2.f);
    // }

    [[nodiscard]] static auto get_offsprings_generation(std::same_as<generation_type> auto... generations)
        -> generation_type
    {
        return static_cast<generation_type>(
            static_cast<float>((generations + ...)) / static_cast<float>(sizeof...(generations))
        );
    }

    auto print() const -> void
    {
        m_Brain.print_net();
        std::cout << "ID: " << m_ID << "\nGeneration " << m_Generation << "\nParents: ";
        for (auto e : m_Parents)
            std::cout << e << ' ';
        std::cout << '\n';
    }
};

} // namespace evolution_agent


#endif // EVOLUTION_AGENT