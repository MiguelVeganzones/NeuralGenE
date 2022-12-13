#pragma once

#ifndef NEURAL_MODEL
#define NEURAL_MODEL

#include <atomic>
#include <concepts>
#include <iostream>
#include <memory>
#include <type_traits>

#include "score_functions.hpp"
#include "static_neural_net.hpp"

namespace ga_neural_model
{

struct score_function_params
{
    score_functions::Identifiers score_function;

    int wins_weight;
    int ties_weight;
    int loses_weight;
};

template <typename NNet>
concept neural_net_type = requires {
                              // Si hubiese más tipos de redes se podrían añadir aqui
                              ga_snn::static_neural_net_type<NNet>;
                          };

template <neural_net_type NNet, score_function_params Score_Function_Params>
class brain
{
public:
    using value_type        = typename NNet::value_type;
    using input_type        = typename NNet::input_type;
    using output_type       = typename NNet::output_type;
    using points_value_type = std::uint16_t;
    using score_value_type  = double;
    using score_functions   = score_functions::score_functions<score_value_type, points_value_type>;
    using f_ptr             = score_functions::f_ptr;

    inline static std::atomic<size_t> s_ID = 1;

    inline static constexpr f_ptr s_score_metric =
        score_functions::choose_function<Score_Function_Params.score_function,
                                         Score_Function_Params.wins_weight,
                                         Score_Function_Params.ties_weight,
                                         Score_Function_Params.loses_weight>();


private:
    size_t            m_ID = s_ID++;
    size_t            m_Parent_a{};
    size_t            m_Parent_b{};
    size_t            m_Generation{};
    points_value_type m_Wins{};
    points_value_type m_Loses{};
    points_value_type m_Ties{};

    std::unique_ptr<NNet> m_Ptr_net{ std::make_unique<NNet>() };

public:
    brain() = default;

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<value_type, Fn, Args...>
    explicit brain(Fn fn, Args... args)
    {
        m_Ptr_net->init(fn, args...);
    }

    explicit brain(const NNet& net) : m_Ptr_net{ std::make_unique<NNet>(net) }
    {
    }

    brain(const brain& other) :
        m_Parent_a{ other.ID() },
        m_Generation{ other.generation() + 1 },
        m_Ptr_net{ std::make_unique<NNet>(*other.get()) }
    {
    }

    brain(brain&&) = default;

    brain& operator=(const brain& other)
    {
        if (this != &other)
        {
            m_Parent_a   = other.m_Parent_a;
            m_Parent_b   = other.m_Parent_b;
            m_Generation = other.m_Generation;
            m_Wins       = other.m_Wins;
            m_Loses      = other.m_Loses;
            m_Ties       = other.m_Ties;
            m_Ptr_net.reset(new NNet(*other.get()));
        }
        return *this;
    }

    brain& operator=(brain&&) = default;
    ~brain()                  = default;

    /* Getters and setters */

    [[nodiscard]] auto ID() const
    {
        return m_ID;
    }

    [[nodiscard]] auto generation() const
    {
        return m_Generation;
    }

    [[nodiscard]] const NNet* get() const
    {
        return m_Ptr_net.get();
    }

    [[nodiscard]] NNet* get_raw()
    {
        return m_Ptr_net.get();
    }

    [[nodiscard]] auto parent_a() const
    {
        return m_Parent_a;
    }

    [[nodiscard]] auto parent_b() const
    {
        return m_Parent_b;
    }

    [[nodiscard]] auto& get_unique()
    {
        return m_Ptr_net;
    }

    [[nodiscard]] auto get_Score() const
    {
        return s_score_metric(m_Wins, m_Ties, m_Loses);
    }

    /* Member functions */

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<value_type, Fn, Args...>
    void init(Fn fn, Args... args)
    {
        m_Ptr_net->init(fn, args...);
    }

    output_type weigh(const input_type& in)
    {
        return m_Ptr_net->forward_pass(in);
    }

    void print_net() const
    {
        m_Ptr_net->print_net();
    }

private:
    void reset()
    {
        m_ID = s_ID++;
        // m_Parent_a{};
        // m_Parent_b{};
        // m_Generation++;
        m_Wins  = 0;
        m_Loses = 0;
        m_Ties  = 0;
    }
};

//--------------------------------------------------------------------------------------//
//  Neural net concept
//--------------------------------------------------------------------------------------//

template <neural_net_type NNet, score_function_params Score_Function_Params>
void nnet_dummy(brain<NNet, Score_Function_Params>)
{
}

template <typename T>
concept brain_type = requires { nnet_dummy(std::declval<T>()); };

//--------------------------------------------------------------------------------------//
//  Utility
//--------------------------------------------------------------------------------------//

template <brain_type Brain>
[[nodiscard]] auto L11_brain_net_distance(const Brain& c4_brain_a, const Brain& c4_brain_b)
{
    return L11_net_distance(c4_brain_a.get(), c4_brain_b.get());
}

template <brain_type Brain>
void in_place_brain_x_crossover(Brain& brain_a, Brain& brain_b)
{
    ga_snn::in_place_net_x_crossover(*brain_a.get_raw(), *brain_b.get_raw());
}

template <brain_type Brain>
[[nodiscard]] std::pair<Brain, Brain> brain_x_crossover(const Brain& brain_a, const Brain& brain_b)
{
    auto [ptr_net1, ptr_net2] = ga_snn::net_x_crossover(*brain_a.get_raw(), *brain_b.get_raw());
    return { brain(ptr_net1), brain(ptr_net2) };
}

template <brain_type Brain>
void to_target_brain_x_crossover(const Brain& parent_a, const Brain& parent_b, Brain& child_a, Brain& child_b)
{
    ga_snn::to_target_net_x_crossover(*parent_a.get(), *parent_b.get(), *child_a.get_raw(), *child_b.get_raw());
}

} // namespace ga_neural_model

#endif // !NEURAL_MODEL
