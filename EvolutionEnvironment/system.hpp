#ifndef AGNET_EVALUATION_SYSTEM
#define AGNET_EVALUATION_SYSTEM

#include "error_handling.hpp"
#include "generics.hpp"
#include <concepts>
#include <ranges>
#include <type_traits>

namespace evaluation_system
{

template <typename Fn>
class system
{
public:
    using fitness_score_type = typename Fn::fitness_score_type;
    using stimulus_type      = typename Fn::stimulus_type;

public:
    system(Fn const& fn) noexcept :
        m_Evaluation_function{ fn }
    {
    }

    template <typename Agent_Type, std::size_t N>
        requires std::is_invocable_r_v<fitness_score_type, Fn, Agent_Type> ||
        std::is_invocable_r_v<
                     std::array<Agent_Type, N>,
                     Fn,
                     std::array<Agent_Type, N>>
    [[nodiscard]]
    auto evaluate(std::array<Agent_Type, N> const& population) const
        -> std::array<fitness_score_type, N>
    {
        if constexpr (std::is_invocable_r_v<fitness_score_type, Fn, Agent_Type>)
        {
            std::array<fitness_score_type, N> ret;
            for (std::size_t i = 0; i != N; ++i)
            {
                ret[i] = std::invoke(m_Evaluation_function, population[i]);
            }
            return ret;
        }
        else if constexpr (std::is_invocable_r_v<
                               std::array<fitness_score_type, N>,
                               Fn,
                               std::array<Agent_Type, N>>)
        {
            return std::invoke(m_Evaluation_function, population);
        }
        else
        {
            assert_unreachable();
        }
    }

    template <typename Agent_Type>
        requires std::is_invocable_r_v<fitness_score_type, Fn, Agent_Type> ||
        std::is_invocable_r_v<
                     std::vector<fitness_score_type>,
                     Fn,
                     std::vector<Agent_Type>>
    [[nodiscard]]
    auto evaluate(std::vector<Agent_Type> const& population) const
        -> std::vector<fitness_score_type>
    {
        if constexpr (std::is_invocable_r_v<fitness_score_type, Fn, Agent_Type>)
        {
            const auto                      n = population.size();
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