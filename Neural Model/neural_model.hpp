#pragma once

#ifndef NEURAL_MODEL
#define NEURAL_MODEL

#include "data_processor.hpp"
#include <atomic>
#include <concepts>
#include <iostream>
#include <memory>
#include <type_traits>

namespace ga_neural_model
{

template <typename NNet>
concept neural_net_concept = requires(NNet net) {
                                 net.print_net();
                                 net.print_address();
                                 net.forward_pass(std::declval<typename NNet::input_type>());
                                 net.mutate(std::declval<typename NNet::value_type (*)(typename NNet::value_type)>());
                                 net.mutate_set_layers(
                                     std::declval<std::vector<std::size_t>>(),
                                     std::declval<typename NNet::value_type (*)(typename NNet::value_type)>()
                                 );
                             };

template <
    neural_net_concept                  NNet,
    data_processor::data_processor_type Data_Preprocessor,
    data_processor::data_processor_type Data_Postprocessor,
    typename Brain_Output_Type>
    requires requires {
                 Data_Postprocessor::template process<typename NNet::output_type, Brain_Output_Type>(
                     std::declval<typename NNet::output_type>()
                 );
             }
class brain
{
public:
    using neural_net_type   = NNet;
    using nn_value_type     = typename NNet::value_type;
    using nn_input_type     = typename NNet::input_type;
    using nn_output_type    = typename NNet::output_type;
    using preprocessor      = Data_Preprocessor;
    using postprocessor     = Data_Postprocessor;
    using brain_output_type = Brain_Output_Type;
    using value_type        = nn_value_type;

    inline static constexpr size_t s_Brain_layers = NNet::s_Layers;


private:
    std::unique_ptr<NNet> m_Ptr_net;

public:
    brain() noexcept = default;

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<nn_value_type, Fn, Args...>
    explicit brain(Fn&& fn, Args&&... args) noexcept :
        m_Ptr_net{ std::make_unique<NNet>() }
    {
        m_Ptr_net->init(std::forward<Fn>(fn), std::forward<Args>(args)...);
    }

    explicit brain(const NNet& net) noexcept :
        m_Ptr_net{ std::make_unique<NNet>(net) }
    {
    }

    brain(const brain& other) noexcept :
        m_Ptr_net{ std::make_unique<NNet>(*other.get()) }
    {
    }

    explicit brain(std::unique_ptr<NNet>&& other_ptr_net) noexcept :
        m_Ptr_net(std::move(other_ptr_net))
    {
    }

    brain(brain&& other) noexcept = default;

    brain& operator=(const brain& other) noexcept
    {
        if (other.m_Ptr_net)
            m_Ptr_net = std::make_unique<NNet>(*other.m_Ptr_net);
        else
            m_Ptr_net.reset();
        return *this;
    }

    brain& operator=(brain&& other) noexcept = default;

    ~brain() noexcept = default;

    /* Getters and setters */

    [[nodiscard]] const NNet& get_net() const
    {
        return *get();
    }

    [[nodiscard]] NNet& get_net()
    {
        return *get_raw();
    }

    [[nodiscard]] const NNet* get() const
    {
        return m_Ptr_net.get();
    }

    [[nodiscard]] NNet* get_raw()
    {
        return m_Ptr_net.get();
    }

    [[nodiscard]] auto& get_unique()
    {
        return m_Ptr_net;
    }

    /* Member functions */

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<nn_value_type, Fn, Args...>
    void init(Fn&& fn, Args&&... args)
    {
        m_Ptr_net->init(std::forward<Fn>(fn), std::forward<Args>(args)...);
    }

    template <typename Brain_Input_Type>
        requires requires {
                     preprocessor::template process<Brain_Input_Type, nn_input_type>(std::declval<Brain_Input_Type>());
                 }
    [[nodiscard]] auto operator()(const Brain_Input_Type& in) const -> brain_output_type
    {
        return postprocessor::template process<nn_output_type, brain_output_type>(
            weigh(preprocessor::template process<Brain_Input_Type, nn_input_type>(in))
        );
    }

    nn_output_type weigh(const nn_input_type& in) const
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
        requires std::is_invocable_r_v<nn_value_type, Fn, nn_value_type>
    void mutate(Fn&& fn)
    {
        m_Ptr_net->mutate(std::forward<Fn>(fn));
    }

    template <typename Fn>
        requires std::is_invocable_r_v<nn_value_type, Fn, nn_value_type>
    void mutate_set_layers(Fn&& fn, const std::vector<size_t>& layers_idx)
    {
        m_Ptr_net->mutate_set_layers(layers_idx, std::forward<Fn>(fn));
    }
};

//--------------------------------------------------------------------------------------//
//  Neural net concept
//--------------------------------------------------------------------------------------//

template <
    neural_net_concept                  NNet,
    data_processor::data_processor_type Data_Preprocessor,
    data_processor::data_processor_type Data_Postprocessor,
    typename Brain_Output_Type>
void brain_dummy(brain<NNet, Data_Preprocessor, Data_Postprocessor, Brain_Output_Type>)
{
}

template <typename T>
concept brain_type = requires(T&& t) { brain_dummy(std::declval<std::remove_cvref_t<T>>()); };

//--------------------------------------------------------------------------------------//
//  Utility
//--------------------------------------------------------------------------------------//

template <std::floating_point R, brain_type Brain, typename Distance>
    requires requires(Brain brain) {
                 {
                     brain.get_net()
                     } -> std::convertible_to<typename std::remove_cvref_t<Brain>::neural_net_type>;
             }
[[nodiscard]] auto distance(Brain const& c4_brain_a, Brain const& c4_brain_b, Distance&& dist_op) -> R
{
    return distance<R>(c4_brain_a.get_net(), c4_brain_b.get_net(), std::forward<Distance>(dist_op));
}

template <brain_type Brain>
[[nodiscard]] auto brain_crossover(const Brain& brain_a, const Brain& brain_b) -> std::pair<Brain, Brain>
{
    auto [ptr_net1, ptr_net2] = net_x_crossover(*brain_a.get(), *brain_b.get());
    return { Brain{ std::move(ptr_net1) }, Brain{ std::move(ptr_net2) } };
}

// TODO Fix the get raw thing
template <brain_type Brain>
auto in_place_brain_crossover(Brain& brain_a, Brain& brain_b) -> void
{
    in_place_net_x_crossover(*brain_a.get_raw(), *brain_b.get_raw());
}

template <brain_type Brain>
auto to_target_brain_crossover(const Brain& parent_a, const Brain& parent_b, Brain& child_a, Brain& child_b) -> void
{
    to_target_net_x_crossover(*parent_a.get(), *parent_b.get(), *child_a.get_raw(), *child_b.get_raw());
}

} // namespace ga_neural_model

#endif // !NEURAL_MODEL