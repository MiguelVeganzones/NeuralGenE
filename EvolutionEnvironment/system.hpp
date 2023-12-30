#ifndef AGNET_EVALUATION_SYSTEM
#define AGNET_EVALUATION_SYSTEM

#include "error_handling.hpp"
#include "generics.hpp"
#include <concepts>
#include <ranges>
#include <type_traits>
#include <vector>

namespace evaluation_system
{

template <typename Fn>
class system
{
public:
    using fitness_score_type = typename Fn::fitness_score_type;
    using stimulus_type      = typename Fn::stimulus_type;

public:
    system(Fn& fn) noexcept :
        m_Evaluation_function{ std::forward<Fn>(fn) }
    {
    }

    template <typename Agent_Type>
        requires std::is_invocable_r_v<fitness_score_type, Fn, Agent_Type> ||
        std::is_invocable_r_v<
                     std::vector<Agent_Type>,
                     Fn,
                     std::vector<Agent_Type>>
    [[nodiscard]]
    auto evaluate(std::vector<Agent_Type> const& population) const
        -> std::vector<fitness_score_type>
    {
        const auto n = population.size();
        if constexpr (std::is_invocable_r_v<fitness_score_type, Fn, Agent_Type>)
        {
            std::vector<fitness_score_type> ret(n);
            for (std::size_t i = 0; i != n; ++i)
            {
                ret[i] = std::invoke(m_Evaluation_function, population[i]);
            }
            return ret;
        }
        else if constexpr (std::is_invocable_r_v<
                               std::vector<fitness_score_type>,
                               Fn,
                               std::vector<Agent_Type>>)
        {
            return std::invoke(m_Evaluation_function, population);
        }
        else
        {
            assert_unreachable();
        }
    }

private:
    Fn m_Evaluation_function;
};
} // namespace evaluation_system


#endif // AGNET_EVALUATION_SYSTEM