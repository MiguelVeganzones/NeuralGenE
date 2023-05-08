#pragma once

#ifndef NEURAL_MODEL
#define NEURAL_MODEL

#include <atomic>
#include <concepts>
#include <iostream>
#include <memory>
#include <type_traits>

#include "data_processor.hpp"
#include "static_neural_net.hpp"

namespace ga_neural_model
{

template <typename NNet>
concept neural_net_type = (
    // More valid neural net types could be added here
    ga_snn::static_neural_net_type<NNet>
);

template <
    neural_net_type                     NNet,
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
    using nn_value_type     = typename NNet::value_type;
    using nn_input_type     = typename NNet::input_type;
    using nn_output_type    = typename NNet::output_type;
    using preprocessor      = Data_Preprocessor;
    using postprocessor     = Data_Postprocessor;
    using brain_output_type = Brain_Output_Type;

    inline static constexpr size_t s_Brain_layers = NNet::s_Layers;


private:
    std::unique_ptr<NNet> m_Ptr_net;

public:
    brain() = default;

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<nn_value_type, Fn, Args...>
    explicit brain(Fn&& fn, Args... args) :
        m_Ptr_net{ std::make_unique<NNet>() }
    {
        m_Ptr_net->init(fn, args...);
    }

    explicit brain(const NNet& net) :
        m_Ptr_net{ std::make_unique<NNet>(net) }
    {
    }

    explicit brain(std::unique_ptr<NNet>&& other_ptr_net) :
        m_Ptr_net{ nullptr }
    {
        std::swap(m_Ptr_net, other_ptr_net);
    }

    /* Getters and setters */

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
    void init(Fn&& fn, Args... args)
    {
        m_Ptr_net->init(fn, args...);
    }

    template <typename Brain_Input_Type>
        requires requires {
                     preprocessor::template process<Brain_Input_Type, nn_input_type>(std::declval<Brain_Input_Type>());
                 }
    auto operator()(const Brain_Input_Type& in) const -> brain_output_type
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
        m_Ptr_net->mutate(fn);
    }

    template <typename Fn>
        requires std::is_invocable_r_v<nn_value_type, Fn, nn_value_type>
    void mutate_set_layers(const std::vector<size_t>& layers_idx, Fn&& fn)
    {
        m_Ptr_net->mutate_set_layers(layers_idx, fn);
    }
};

//--------------------------------------------------------------------------------------//
//  Neural net concept
//--------------------------------------------------------------------------------------//

template <
    neural_net_type                     NNet,
    data_processor::data_processor_type Data_Preprocessor,
    data_processor::data_processor_type Data_Postprocessor,
    typename Brain_Output_Type>
void nnet_dummy(brain<NNet, Data_Preprocessor, Data_Postprocessor, Brain_Output_Type>)
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
