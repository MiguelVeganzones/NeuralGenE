#pragma once

#ifndef NEURAL_MODEL
#define NEURAL_MODEL

#include <concepts>
#include <iostream>
#include <memory>
#include <type_traits>

#include "../Static Neural Net/static_neural_net.hpp"

namespace ga_c4_brain_v2
{

template <typename NNet>
concept neural_net_type = requires {
                              ga_snn::static_neural_net_type<NNet>; // Si hubiese más tipos de
                                                                    // redes se podrían añadir aqui
                          };

template <neural_net_type NNet>
class c4_brain
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
    c4_brain() = default;

    explicit c4_brain(const NNet& net) : m_Ptr_net{ std::make_unique<NNet>(net) }
    {
    }

    c4_brain(const c4_brain& other) :
        m_Parent_a{ other.ID() }, m_Generation{ other.generation() + 1 }, m_Ptr_net{ std::make_unique<NNet>(
                                                                              *other.get()) }
    {
    }

    c4_brain(c4_brain&&) = default;

    c4_brain& operator=(const c4_brain& other)
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

    c4_brain& operator=(c4_brain&&) = default;
    ~c4_brain()                     = default;

    /* Getters and setters */

    auto ID() const
    {
        return m_ID;
    }
    auto generation() const
    {
        return m_Generation;
    }
    const NNet* get() const
    {
        return m_Ptr_net.get();
    }
    auto parent_a() const
    {
        return m_Parent_a;
    }
    auto parent_b() const
    {
        return m_Parent_b;
    }
    auto& get_unique()
    {
        return m_Ptr_net;
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
[[nodiscard]] auto L11_brain_net_distance(const c4_brain<NNet>& c4_brain_a, const c4_brain<NNet>& c4_brain_b)
{
    return L11_net_distance(c4_brain_a.get(), c4_brain_b.get());
}

} // namespace ga_c4_brain_v2

#endif // !NEURAL_MODEL
