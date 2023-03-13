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

template <typename NNet>
concept neural_net_type = requires {
                              // Si hubiese más tipos de redes se podrían añadir aqui
                              ga_snn::static_neural_net_type<NNet>;
                          };

template <neural_net_type NNet, score_function_objects::score_function_object_type Score_Function>
class brain
{
public:
    using value_type  = typename NNet::value_type;
    using input_type  = typename NNet::input_type;
    using output_type = typename NNet::output_type;

    inline static constexpr size_t s_Brain_layers = NNet::s_Layers;

    inline static std::atomic<size_t> s_ID = 1;

private:
    size_t m_ID = s_ID++;
    size_t m_Parent_a{};
    size_t m_Parent_b{};
    size_t m_Generation{};

    std::unique_ptr<NNet>           m_Ptr_net;
    std::unique_ptr<Score_Function> m_Ptr_score_function;

public:
    brain() = default;

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<value_type, Fn, Args...>
    explicit brain(Fn&& fn, Args... args) : m_Ptr_net{ std::make_unique<NNet>() }
    {
        m_Ptr_net->init(fn, args...);
    }

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<value_type, Fn, Args...>
    explicit brain(Score_Function score_function, Fn&& fn, Args... args) :
        m_Ptr_net{ std::make_unique<NNet>() }, m_Ptr_score_function{ std::make_unique<Score_Function>(score_function) }
    {
        m_Ptr_net->init(fn, args...);
    }

    explicit brain(const NNet& net) : m_Ptr_net{ std::make_unique<NNet>(net) }
    {
    }

    explicit brain(std::unique_ptr<NNet>&& other_ptr_net) : m_Ptr_net{ nullptr }
    {
        std::swap(m_Ptr_net, other_ptr_net);
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
            m_Ptr_net.reset(new NNet(*other.get()));
        }
        return *this;
    }

    brain& operator=(brain&&) = default;
    ~brain()                  = default;

    /* Getters and setters */

    void set_score_function_obj(Score_Function score_function)
    {
        m_Ptr_score_function = score_function;  
    }

    [[nodiscard]] Score_Function* get_score_function_obj()
    {
        return m_Ptr_score_function.get();  
    }

    [[nodiscard]] auto get_score()
    {
        return m_Ptr_score_function->operator()();  
    }

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

    [[nodiscard]] auto get_score() const
    {
        if (m_Ptr_score_function)
            return m_Ptr_score_function->operator()();

        throw std::exception("m_Function_ptr is not set.");
    }

    /* Member functions */

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<value_type, Fn, Args...>
    void init(Fn&& fn, Args... args)
    {
        m_Ptr_net->init(fn, args...);
    }

    output_type weigh(const input_type& in) const
    {
        return m_Ptr_net->forward_pass(in);
    }

    value_type weigh(value_type in) const
    {
        return m_Ptr_net->forward_pass(in);
    }

    /* Utility */

    void print_net() const
    {
        m_Ptr_net->print_net();
    }

    void print_net_address() const
    {
        m_Ptr_net->print_address();
    }

    /* GA Utility */

    template <typename Fn>
        requires std::is_invocable_r_v<value_type, Fn, value_type>
    void mutate(Fn&& fn)
    {
        m_Ptr_net->mutate(fn);
    }

    template <typename Fn>
        requires std::is_invocable_r_v<value_type, Fn, value_type>
    void mutate_set_layers(const std::vector<size_t>& layers_idx, Fn&& fn)
    {
        m_Ptr_net->mutate_set_layers(layers_idx, fn);
    }

private:
    void reset()
    {
        m_ID = s_ID++;
        // m_Parent_a{};
        // m_Parent_b{};
        // m_Generation++;
    }
};

//--------------------------------------------------------------------------------------//
//  Neural net concept
//--------------------------------------------------------------------------------------//

template <neural_net_type NNet, score_function_objects::score_function_object_type Score_Function>
void nnet_dummy(brain<NNet, Score_Function>)
{
}

template <typename T>
concept brain_type = requires { nnet_dummy(std::declval<T>()); };

//--------------------------------------------------------------------------------------//
//  Utility
//--------------------------------------------------------------------------------------//

template <brain_type Brain>
[[nodiscard]] auto L1_brain_net_distance(const Brain& c4_brain_a, const Brain& c4_brain_b)
{
    return L1_net_distance(c4_brain_a.get(), c4_brain_b.get());
}

template <brain_type Brain>
void in_place_brain_x_crossover(Brain& brain_a, Brain& brain_b)
{
    ga_snn::in_place_net_x_crossover(*brain_a.get_raw(), *brain_b.get_raw());
}

template <brain_type Brain>
[[nodiscard]] std::pair<Brain, Brain> brain_x_crossover(const Brain& brain_a, const Brain& brain_b)
{
    auto [ptr_net1, ptr_net2] = ga_snn::net_x_crossover(*brain_a.get(), *brain_b.get());
    return { Brain{ std::move(ptr_net1) }, Brain{ std::move(ptr_net2) } };
}

template <brain_type Brain>
void to_target_brain_x_crossover(const Brain& parent_a, const Brain& parent_b, Brain& child_a, Brain& child_b)
{
    ga_snn::to_target_net_x_crossover(*parent_a.get(), *parent_b.get(), *child_a.get_raw(), *child_b.get_raw());
}

} // namespace ga_neural_model

#endif // !NEURAL_MODEL
