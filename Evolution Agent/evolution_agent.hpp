#ifndef EVOLUTION_AGENT
#define EVOLUTION_AGENT

#include "Random.h"
#include "data_processor.hpp"
#include "neural_model.hpp"
#include "score_functions.hpp"
#include <iostream>

namespace evolution_agent
{

template <ga_neural_model::brain_type Brain, score_function_objects::score_function_object_type Score_Function>
class agent
{
public:
    using score_function_type    = Score_Function;
    using brain_type             = Brain;
    using id_type                = std::size_t;
    using generation_type        = std::size_t;
    using parents_container_type = std::vector<id_type>;

private:
    inline static std::atomic<id_type> s_ID = 0;

    id_type                m_ID = s_ID++;
    brain_type             m_Brain{};
    score_function_type    m_Score_function{};
    parents_container_type m_Parents{};
    generation_type        m_Generation{};


private:
    agent(
        const brain_type&             brain,
        const score_function_type&    score_function,
        const parents_container_type& parents,
        generation_type               generation
    ) :
        m_Brain(brain),
        m_Score_function{ score_function },
        m_Parents{ parents },
        m_Generation{ generation }
    {
    }

public:
    explicit agent(auto brain_initializer)
        requires requires { brain_type(brain_initializer); }
        :
        m_Brain(brain_initializer)
    {
    }

    /// @brief Asexual reproduction by cloning. Generates an identical agent member of the next generation.
    /// @return New Agent member of the next generation to the parent.
    agent clone() const
    {
        return agent(m_Brain, m_Score_function, { m_ID }, m_Generation + 1);
    }

    // [[nodiscard]] auto generation() const
    // {
    //     return m_Generation;
    // }
};

} // namespace evolution_agent


#endif // EVOLUTION_AGENT