#ifndef EVOLUTION_ENVIRONMENT_TRAITS
#define EVOLUTION_ENVIRONMENT_TRAITS

#include <array>
#include <concepts>
#include <type_traits>

namespace evolution_environment_traits
{

template <typename T>
concept agent_concept = requires(T t) {
    typename T::brain_value_type;
    // typename T::agent_input_type;
    typename T::agent_output_type;
    // {
    //     t.mutate()
    // } -> std::same_as<void>;
    {
        to_target_crossover(t, t, t, t)
    } -> std::same_as<void>;
    // {
    //     t(std::declval<typename T::agent_input_type>)
    // } -> std::same_as<typename T::agent_output_type>;
};

template <typename T>
concept system_concept = requires(T t) {
    typename T::stimulus_type;
    typename T::fitness_score_type;
    // FIXME in the future
    // require system to be able to evaluate a range of agents. Cannot do it
    // with current concepts without using an archetype agent
};

template <typename T>
concept reproduction_manager_concept = requires(T t) {
    typename T::agent_type;
    {
        t.yield_next_generation(
            std::declval<typename T::generation_container_type&>(),
            std::declval<typename T::generation_fitness_container_type&>(),
            std::declval<typename T::generation_container_type&>()
        )
    } -> std::same_as<void>;
};

} // namespace evolution_environment_traits

#endif // EVOLUTION_ENVIRONMENT_TRAITS