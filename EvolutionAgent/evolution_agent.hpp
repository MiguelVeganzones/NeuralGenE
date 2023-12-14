#ifndef EVOLUTION_AGENT
#define EVOLUTION_AGENT

#include "Random.hpp"
#include "data_processor.hpp"
#include "generics.hpp"
#include <atomic>
#include <iostream>

namespace evolution_agent
{

template <typename T>
concept brain_concept = requires(T t) {
    typename T::brain_output_type;
    typename T::value_type;
    T();
    t.mutate(std::declval<typename T::value_type (*)(typename T::value_type)>()
    );
    {
        to_target_brain_crossover(t, t, t, t)
    } -> std::same_as<void>;
};

template <typename T>
concept mutation_policy_concept = requires(T t) {
    typename T::value_type;
    typename T::parameters_type;
    {
        t(std::declval<typename T::value_type&>())
    } -> std::same_as<typename T::value_type>;
};

template <brain_concept Brain, mutation_policy_concept Mutation_Policy_Type>
class agent
{
public:
    using brain_type        = std::remove_cvref_t<Brain>;
    using brain_value_type  = typename brain_type::value_type;
    using agent_output_type = typename brain_type::brain_output_type;
    // using agent_input_type       = typename brain_type::brain_input_type;
    using id_type                = std::size_t;
    using generation_type        = std::size_t;
    using parents_container_type = std::vector<id_type>; // TODO: Optimize(?)
    using mutation_policy_type   = Mutation_Policy_Type;

private:
    inline static std::atomic<id_type> s_ID = 1;

    id_type                m_ID = s_ID++;
    generation_type        m_Generation{};
    parents_container_type m_Parents{};
    brain_type             m_Brain{};
    mutation_policy_type   m_Mutation_policy{};


public:
    agent(
        const brain_type&             brain,
        const parents_container_type& parents,
        generation_type               generation,
        mutation_policy_type          mutation_policy
    ) noexcept :
        m_Generation{ generation },
        m_Parents{ parents },
        m_Brain(brain),
        m_Mutation_policy{ mutation_policy }
    {
    }

    agent(
        brain_type&&                  brain,
        const parents_container_type& parents,
        generation_type               generation,
        mutation_policy_type          mutation_policy
    ) noexcept :
        m_Generation{ generation },
        m_Parents{ parents },
        m_Brain(std::move(brain)),
        m_Mutation_policy{ mutation_policy }
    {
    }

    agent(Brain&& brain, mutation_policy_type mutation_policy) noexcept :
        m_Brain(std::forward<Brain>(brain)),
        m_Mutation_policy{ mutation_policy }
    {
    }

    agent() noexcept                        = delete;
    agent(const agent&) noexcept            = default;
    agent(agent&&) noexcept                 = default;
    agent& operator=(const agent&) noexcept = default;
    agent& operator=(agent&&) noexcept      = default;
    ~agent() noexcept                       = default;

    //--------------------------------------------------------------------------------------//
    // Agent asexual reproduction
    //--------------------------------------------------------------------------------------//

    /// @brief Asexual reproduction by cloning. Generates an identical agent
    /// member of the next generation.
    /// @return New Agent member of the next generation to the parent.
    [[nodiscard]]
    auto clone() const -> agent
    {
        return agent(m_Brain, { m_ID }, m_Generation + 1, m_Mutation_policy);
    }

    void mutate()
        requires std::is_invocable_r_v<
            brain_value_type,
            mutation_policy_type,
            brain_value_type>
    {
        m_Brain.mutate(m_Mutation_policy);
    }

    //--------------------------------------------------------------------------------------//
    // Agent interface with its environment
    //--------------------------------------------------------------------------------------//

    template <typename Agent_Input_Type>
        requires requires { m_Brain(std::declval<Agent_Input_Type&>()); }
    [[nodiscard]]
    auto
    operator()(const Agent_Input_Type& input) const -> agent_output_type
    {
        return m_Brain(input);
    }

    //--------------------------------------------------------------------------------------//
    // Utility
    //--------------------------------------------------------------------------------------//


    [[nodiscard]]
    auto get_ID() const -> id_type
    {
        return m_ID;
    }

    [[nodiscard]]
    auto get_generation() const -> generation_type
    {
        return m_Generation;
    }

    [[nodiscard]]
    auto get_brain() const -> const brain_type&
    {
        return m_Brain;
    }

    [[nodiscard]]
    auto get_brain() -> brain_type&
    {
        return m_Brain;
    }

    [[nodiscard]]
    auto get_mutation_policy() const -> mutation_policy_type
    {
        return m_Mutation_policy;
    }

    // [[nodiscard]] static auto get_offsprings_generation(generation_type
    // gen_a, generation_type gen_b) -> generation_type
    // {
    //     return static_cast<generation_type>(static_cast<float>(gen_a + gen_b)
    //     / 2.f);
    // }

    [[nodiscard]]
    static auto get_offsprings_generation(
        std::same_as<generation_type> auto... generations
    ) -> generation_type
    {
        return static_cast<generation_type>(
            static_cast<float>((generations + ...)) /
            static_cast<float>(sizeof...(generations))
        );
    }

    auto print() const -> void
    {
        m_Brain.print_net();
        std::cout << "ID: " << m_ID << "\nGeneration " << m_Generation
                  << "\nParents: ";
        for (auto e : m_Parents)
            std::cout << e << ' ';
        std::cout << "\nMutation policy parameters: [ ";
        for (auto e : m_Mutation_policy.get_parameters())
            std::cout << e << ' ';
        std::cout << "]\n";
    }
};

//--------------------------------------------------------------------------------------//
//  Neural net concept
//--------------------------------------------------------------------------------------//

template <brain_concept Brain, mutation_policy_concept Mutation_Policy>
void agent_dummy(agent<Brain, Mutation_Policy>)
{
}

template <typename T>
concept agent_type =
    requires { agent_dummy(std::declval<std::remove_cvref_t<T>&>()); };

//--------------------------------------------------------------------------------------//
// Agent sexual reproduction
//--------------------------------------------------------------------------------------//

template <agent_type Agent>
auto to_target_crossover(
    const Agent& parent_a,
    const Agent& parent_b,
    Agent&       child_a,
    Agent&       child_b
) -> void
{
    to_target_brain_crossover(
        parent_a.get_brain(),
        parent_b.get_brain(),
        child_a.get_brain(),
        child_b.get_brain()
    );
}

template <std::floating_point R, agent_type Agent, typename Distance>
    requires requires(Agent agent) {
        {
            agent.get_brain()
        } -> std::convertible_to<typename Agent::brain_type>;
    }
[[nodiscard]]
auto agent_distance(
    Agent const& agent_a,
    Agent const& agent_b,
    Distance&&   dist_op
) -> R
{
    return brain_distance<R>(
        agent_a.get_brain(),
        agent_b.get_brain(),
        std::forward<Distance>(dist_op)
    );
}

// TODO change to mdspan when gcc implements it (clang has support for it
// already)

template <
    std::floating_point R,
    agent_type          Agent,
    std::size_t         N,
    typename Distance>
    requires(N > 1)
[[nodiscard]]
static auto population_variability(
    std::array<Agent, N> const& agents,
    Distance&&                  dist_op
) -> ga_sm::static_matrix<R, N, N>
{
    ga_sm::static_matrix<R, N, N> distance_matrix{};
    for (size_t j = 0; j != N - 1; ++j)
    {
        for (size_t i = j + 1; i != N; ++i)
        {
            const auto distance =
                agent_distance<R>(agents[j], agents[i], dist_op);
            distance_matrix[j, i] = distance;
            distance_matrix[i, j] = distance;
        }
    }
    return distance_matrix;
}

} // namespace evolution_agent


#endif // EVOLUTION_AGENT