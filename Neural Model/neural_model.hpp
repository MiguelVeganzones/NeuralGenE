#pragma once

#ifndef NEURAL_MODEL
#define NEURAL_MODEL

#include <concepts>
#include <iostream>
#include <memory>
#include <type_traits>

#include "static_neural_net.hpp"

namespace ga_neural_model
{

template <typename NNet>
concept neural_net_type = requires {
                              // Si hubiese más tipos de redes se podrían añadir aqui
                              ga_snn::static_neural_net_type<NNet>;
                          };

template <neural_net_type NNet>
class brain
{
public:
    using value_type  = typename NNet::value_type;
    using input_type  = typename NNet::input_type;
    using output_type = typename NNet::output_type;

    inline static size_t s_ID = 1;

private:
    size_t        m_ID = s_ID++;
    double        m_Score{};
    size_t        m_Parent_a{};
    size_t        m_Parent_b{};
    size_t        m_Generation{};
    std::uint16_t m_Wins{};
    std::uint16_t m_Loses{};
    std::uint16_t m_Ties{};

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
            m_Score      = other.m_Score;
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

    void print_net() const
    {
        m_Ptr_net->print_net();
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
};

/* Utility */

template <neural_net_type NNet>
[[nodiscard]] auto L11_brain_net_distance(const brain<NNet>& c4_brain_a, const brain<NNet>& c4_brain_b)
{
    return L11_net_distance(c4_brain_a.get(), c4_brain_b.get());
}

template <neural_net_type NNet>
void in_place_brain_x_crossover(brain<NNet>& brain_a, brain<NNet>& brain_b)
{
    ga_snn::in_place_net_x_crossover(*brain_a.get_raw(), *brain_b.get_raw());
}

template <neural_net_type NNet>
[[nodiscard]] std::pair<brain<NNet>, brain<NNet>> brain_x_crossover(brain<NNet>& brain_a, brain<NNet>& brain_b)
{
    auto [ptr_net1, ptr_net2] = ga_snn::net_x_crossover(*brain_a.get_raw(), *brain_b.get_raw());
    return { brain(ptr_net1), brain(ptr_net2) };
}

} // namespace ga_neural_model

#endif // !NEURAL_MODEL
