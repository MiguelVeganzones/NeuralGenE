#ifndef STATIC_NEURAL_NET
#define STATIC_NEURAL_NET

#include <array>
#include <concepts>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

#include "Log.h"
#include "activation_functions.hpp"
#include "static_matrix.hpp"

namespace ga_snn
{

struct Layer_Signature
{
    size_t                                                 Size;
    matrix_activation_functions::Identifiers::Identifiers_ Activation;

    constexpr bool operator==(const Layer_Signature&) const = default;
};

struct Layer_Structure
{
    size_t                                                 Inputs;
    size_t                                                 Outputs;
    matrix_activation_functions::Identifiers::Identifiers_ Activation;

    constexpr bool operator==(const Layer_Structure&) const = default;
};

//--------------------------------------------------------------------------------------//

template <std::floating_point T, size_t Batch_Size, Layer_Structure Structure>
    requires(Batch_Size > 0)
class layer
{
public:
    static constexpr size_t s_Inputs     = Structure.Inputs;
    static constexpr size_t s_Outputs    = Structure.Outputs;
    static constexpr auto   s_Activation = Structure.Activation;

    using weights_shape       = ga_sm::static_matrix<T, s_Inputs, s_Outputs>;
    using output_vector_shape = ga_sm::static_matrix<T, Batch_Size, s_Outputs>;
    using input_vector_shape  = ga_sm::static_matrix<T, Batch_Size, s_Inputs>;
    using bias_vector_shape   = ga_sm::static_matrix<T, 1, s_Outputs>;
    using activation_function =
        matrix_activation_functions::activation_function<output_vector_shape, Structure.Activation>;

private:
    weights_shape       m_weights_mat;
    bias_vector_shape   m_bias_vector;
    activation_function m_activation_function;

public:
    // each neuron produces a column vector that splits into the input of the
    // neurons in the next layer
    [[nodiscard]] constexpr output_vector_shape forward_pass(input_vector_shape const& Input) const
    {
        auto out = matrix_vec_add(matrix_mul(Input, m_weights_mat), m_bias_vector);
        // std::cout << out << std::endl;
        m_activation_function(out);
        // std::cout << out << std::endl;
        return out;
    }

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<T, Fn, Args...>
    constexpr void init(Fn&& fn, Args&&... args)
    {
        m_weights_mat.fill(fn, std::forward<Args>(args)...);
        m_bias_vector.fill(fn, std::forward<Args>(args)...);
        m_activation_function.fill(fn, std::forward<Args>(args)...);
    }

    template <typename Fn>
    constexpr void mutate(Fn&& fn)
    {
        m_weights_mat.transform(fn);
        m_bias_vector.transform(fn);
        m_activation_function.mutate_params(fn);
    }

    void store(std::ofstream& out) const
    {
        m_weights_mat.store(out);
        m_bias_vector.store(out);
        m_activation_function.store(out);
    }

    void load(std::ifstream& in)
    {
        m_weights_mat.load(in);
        m_bias_vector.load(in);
        m_activation_function.load(in);
    }

    [[nodiscard]] inline constexpr const weights_shape& get_weights_mat() const
    {
        return m_weights_mat;
    }

    [[nodiscard]] inline constexpr weights_shape& get_weights_mat()
    {
        return m_weights_mat;
    }

    [[nodiscard]] inline constexpr const bias_vector_shape& get_bias_vector() const
    {
        return m_bias_vector;
    }

    [[nodiscard]] inline constexpr bias_vector_shape& get_bias_vector()
    {
        return m_bias_vector;
    }

    [[nodiscard]] inline constexpr const activation_function& get_activation_function() const
    {
        return m_activation_function;
    }

    [[nodiscard]] inline constexpr activation_function& get_activation_function()
    {
        return m_activation_function;
    }

    [[nodiscard]] static inline constexpr size_t get_inputs()
    {
        return s_Inputs;
    }

    [[nodiscard]] static inline constexpr size_t get_outputs()
    {
        return s_Outputs;
    }

    [[nodiscard]] static inline constexpr size_t parameter_count()
    {
        return s_Outputs * (s_Inputs + 1) + activation_function::parameter_count();
    }

    [[nodiscard]] static inline constexpr size_t layer_size()
    {
        return sizeof(layer);
    }
};

// -----------------------------------------------
// Static layer concept
// -----------------------------------------------

template <typename T, size_t Batch_Size, Layer_Structure Structure>
void layer_dummy(layer<T, Batch_Size, Structure>)
{
}

template <typename T>
concept static_layer_type = requires { layer_dummy(std::declval<T>()); };

// -----------------------------------------------
// -----------------------------------------------

template <static_layer_type Layer>
std::ostream& operator<<(std::ostream& os, const Layer& layer)
{
    os << layer.get_weights_mat() << layer.get_bias_vector() << '\n' << layer.get_activation_function() << '\n';
    return os;
}

//-------------------------------------------------------------------------------
//
//-------------------------------------------------------------------------------

// base template
template <typename T, size_t Inputs, size_t Batch_Size, Layer_Signature...>
    requires(Batch_Size > 0)
struct layer_unroll;

// specialization for last layer
template <typename T, size_t Inputs, size_t Batch_Size, Layer_Signature Current_Signature>
    requires(Batch_Size > 0)
struct layer_unroll<T, Inputs, Batch_Size, Current_Signature>
{
    static constexpr size_t s_Inputs{ Inputs };
    static constexpr size_t s_Outputs{ Current_Signature.Size };

    using current_layer_type =
        layer<T, Batch_Size, Layer_Structure{ s_Inputs, s_Outputs, Current_Signature.Activation }>;

    current_layer_type m_Data; // one data member for this layer

    template <size_t Idx>
    [[nodiscard]] constexpr auto& get()
    {
        if constexpr (Idx == 0)
        { // if its 0, return this layer
            return m_Data;
        }
        else
        {
            static_assert(false && Idx, "requested layer index exceeds number of layers");
        }
    }

    template <size_t Idx>
    [[nodiscard]] constexpr auto const& const_get() const
    {
        if constexpr (Idx == 0)
        { // if its 0, return this layer
            return m_Data;
        }
        else
        {
            static_assert(false && Idx, "requested layer index exceeds number of layers.");
        }
    }

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<T, Fn, Args...>
    void init(Fn&& fn, Args&&... args)
    {
        m_Data.init(fn, std::forward<Args>(args)...);
    }

    template <typename Fn>
    void mutate(Fn&& fn)
    {
        m_Data.mutate(fn);
    }

    template <typename Fn>
    void mutate_layer(size_t layer_idx, Fn&& fn)
    {
        layer_idx == 0
            ? m_Data.mutate(fn)
            : throw std::invalid_argument("Invalid layer index: " + std::to_string(layer_idx) + " in last layer.");
    }

    void print() const
    {
        std::cout << m_Data << '\n';
    }

    void store(std::ofstream& out) const
    {
        m_Data.store(out);
    }

    void load(std::ifstream& in)
    {
        m_Data.load(in);
    }

    [[nodiscard]] auto forward_pass(ga_sm::static_matrix<T, Batch_Size, s_Inputs> const& input_data) const
    {
        return m_Data.forward_pass(input_data);
    }

    [[nodiscard]] static size_t layer_size(const int idx_to_target_layer) noexcept
    {
        if (idx_to_target_layer == 0)
        {
            return current_layer_type::layer_size();
        }
        std::unreachable();
    }

    [[nodiscard]] static size_t parameter_count(const int idx_to_target_layer) noexcept
    {
        if (idx_to_target_layer == 0)
        {
            return current_layer_type::parameter_count();
        }
        std::unreachable();
    }
};

// general specialization
template <
    typename T,
    size_t          Inputs,
    size_t          Batch_Size,
    Layer_Signature Current_Signature,
    Layer_Signature... Signatures>
    requires(Batch_Size > 0)
struct layer_unroll<T, Inputs, Batch_Size, Current_Signature, Signatures...>
{
    static constexpr size_t s_Inputs{ Inputs };
    static constexpr size_t s_Outputs{ Current_Signature.Size };

    using current_layer_type =
        layer<T, Batch_Size, Layer_Structure{ s_Inputs, s_Outputs, Current_Signature.Activation }>;
    using next_data_type = layer_unroll<T, Current_Signature.Size, Batch_Size, Signatures...>;

    current_layer_type m_Data; // one data member for this layer
    next_data_type     m_Next; // another layer_unroll member for the rest

    // getter so that we can actually get a specific layer by index
    template <size_t Idx>
    [[nodiscard]] constexpr auto& get() noexcept
    {
        if constexpr (Idx == 0)
        { // if its 0, return this layer
            return m_Data;
        }
        else
        {
            return m_Next.template get<Idx - 1>(); // if the index is not 0, ask the
            // next layer with an updated index
        }
    }

    template <size_t Idx>
    [[nodiscard]] constexpr auto const& const_get() const noexcept
    {
        if constexpr (Idx == 0)
        { // if its 0, return this layer
            return m_Data;
        }
        else
        {
            return m_Next.template const_get<Idx - 1>(); // if the index is not 0, ask the
            // next layer with an updated index
        }
    }

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<T, Fn, Args...>
    void init(Fn&& fn, Args&&... args)
    {
        m_Data.init(fn, std::forward<Args>(args)...);
        m_Next.init(fn, std::forward<Args>(args)...);
    }

    template <typename Fn>
    void mutate(Fn&& fn)
    {
        m_Data.mutate(fn);
        m_Next.mutate(fn);
    }

    template <typename Fn>
    void mutate_layer(size_t layer_idx, Fn&& fn)
    {
        layer_idx == 0 ? m_Data.mutate(fn) : m_Next.mutate_layer(--layer_idx, fn);
    }

    void print() const
    {
        std::cout << m_Data << '\n';
        m_Next.print();
    }

    void store(std::ofstream& out) const
    {
        m_Data.store(out);
        m_Next.store(out);
    }

    void load(std::ifstream& in)
    {
        m_Data.load(in);
        m_Next.load(in);
    }

    [[nodiscard]] auto forward_pass(ga_sm::static_matrix<T, Batch_Size, s_Inputs> const& input_data) const
    {
        return m_Next.forward_pass(m_Data.forward_pass(input_data));
    }

    [[nodiscard]] static size_t layer_size(const int idx_to_target_layer)
    {
        if (idx_to_target_layer == 0)
        {
            return current_layer_type::layer_size();
        }
        return next_data_type::layer_size(idx_to_target_layer - 1);
    }

    [[nodiscard]] static size_t parameter_count(const int idx_to_target_layer)
    {
        if (idx_to_target_layer == 0)
        {
            return current_layer_type::parameter_count();
        }
        return next_data_type::parameter_count(idx_to_target_layer - 1);
    }
};

//--------------------------------------------------------------------------------------//

template <std::floating_point T, size_t Batch_Size, Layer_Signature... Signatures>
    requires((sizeof...(Signatures) >= 3) && Batch_Size > 0)
class static_neural_net
{
public:
    static constexpr size_t                                s_Layers{ sizeof...(Signatures) };
    static constexpr std::array<Layer_Signature, s_Layers> s_Signatures{ Signatures... };
    static constexpr size_t                                s_Output_Size = s_Signatures[s_Layers - 1].Size;
    static constexpr size_t                                s_Input_Size  = s_Signatures[0].Size;

private:
    using layers_type = layer_unroll<T, s_Input_Size, Batch_Size, Signatures...>;

    layers_type m_Layers;

public:
    using value_type  = T;
    using input_type  = ga_sm::static_matrix<T, Batch_Size, s_Input_Size>;
    using output_type = ga_sm::static_matrix<T, Batch_Size, s_Output_Size>;

public:
    [[nodiscard]] static constexpr size_t parameter_count(
        unsigned int first_layer_idx = 0, const unsigned int last_layer_idx = s_Layers - 1
    )
    {
        assert(last_layer_idx < s_Layers);
        assert(first_layer_idx <= s_Layers);

        size_t p{};
        for (; first_layer_idx != last_layer_idx + 1; ++first_layer_idx)
        {
            p += layer_parameter_count(first_layer_idx);
        }
        return p;
    }

    [[nodiscard]] static constexpr size_t layer_parameter_count(const unsigned int layer_idx)
    {
        assert(layer_idx < s_Layers);

        return layers_type::parameter_count(layer_idx);
    }

    [[nodiscard]] static constexpr size_t subnet_size(
        unsigned int first_layer_idx = 0, const unsigned int last_layer_idx = s_Layers - 1
    )
    {
        size_t p{};
        for (; first_layer_idx != last_layer_idx + 1; ++first_layer_idx)
        {
            p += layer_size(first_layer_idx);
        }
        return p;
    }

    [[nodiscard]] static constexpr size_t layer_size(const unsigned int layer_idx)
    {
        return layers_type::layer_size(layer_idx);
    }

    template <std::size_t Idx>
    [[nodiscard]] auto& layer()
    {
        return m_Layers.template get<Idx>();
    }

    template <std::size_t Idx>
    [[nodiscard]] auto const& const_layer() const
    {
        return m_Layers.template const_get<Idx>();
    }

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<T, Fn, Args...>
    void init(Fn&& fn, Args&&... args)
    {
        m_Layers.init(fn, std::forward<Args>(args)...);
    }

    template <typename Fn>
    void mutate(Fn&& fn)
    {
        m_Layers.mutate(fn);
    }

    template <typename Fn>
    void mutate_layer(size_t layer_idx, Fn&& fn)
    {
        m_Layers.mutate_layer(layer_idx, fn);
    }

    template <typename Fn>
    void mutate_set_layers(const std::vector<size_t>& layers_idx, Fn&& fn)
    {
        mutate_set_layers_impl(layers_idx, fn);
    }

    template <size_t I = 0, typename Fn>
    void mutate_set_layers_impl(const std::vector<size_t>& layers_idx, Fn&& fn)
    {
        if (std::ranges::find(layers_idx, I) != layers_idx.end())
        {
            this->template layer<I>().mutate(fn);
        }
        if constexpr (I < (s_Layers - 1))
        {
            mutate_set_layers_impl<I + 1>(layers_idx, fn);
        }
    }

    void print_layers() const
    {
        std::ios_base::sync_with_stdio(false);
        m_Layers.print();
        std::ios_base::sync_with_stdio(true);
    }

    void print_net() const
    {
        std::cout << "##########################################################"
                     "################\n";
        print_layers();
        std::cout << "Net has " << parameter_count() << " parameters\n";
        std::cout << "Net size: " << subnet_size() << " bytes\n";
        print_address();
        std::cout << "----------------------------------------------------------"
                     "----------------\n";
    }

    void print_address() const
    {
        std::cout << "Net address: " << this << '\n';
    }

    void store(const std::filesystem::path& filename) const
    {
        std::ofstream out(filename);
        if (!out.is_open())
        {
            const auto message = "Could not create file: " + filename.string() + '\n';
            std::cout << message;
            log::add(message);
            std::exit(EXIT_FAILURE);
        }
        // store net shapes
        for (const auto& layer_signature : s_Signatures)
        {
            out << layer_signature.Size << ' ';
        }
        out << "\n\n";
        m_Layers.store(out); // Store layers recursively
        out.close();
        if (!out)
        {
            const auto message = "Could not close file: " + filename.string() + '\n';
            std::cout << message;
            log::add(message);
            std::exit(EXIT_FAILURE);
        }
    }

    // Overrides current net with one read from "filename"
    // Shapes of both nets must be the same
    void load(const std::filesystem::path& filename)
    {
        std::ifstream in(filename);
        if (!in.is_open())
        {
            const std::string message = "Could not open file: " + filename.string() + '\n';
            std::cout << message;
            log::add(message);
            std::exit(EXIT_FAILURE);
        }
        size_t layer_size{}; // shape
        for (const auto& layer_signature : s_Signatures)
        {
            in >> layer_size;
            if (layer_size != layer_signature.Size)
            {
                const auto message = "Cannot load this net here. Shapes must match.\n";
                std::cout << message;
                log::add(message);
                std::exit(EXIT_FAILURE);
            }
        }
        m_Layers.load(in); // load layers recursively
        in.close();
        if (!in)
        {
            const auto message = "Could not close file after reading: " + filename.string() + '\n';
            std::cout << message;
            log::add(message);
        }
    }

    [[nodiscard]] output_type batch_forward_pass(input_type const& input_data) const
    {
        return m_Layers.forward_pass(input_data);
    }

    [[nodiscard]] output_type forward_pass(input_type const& input_data) const
    {
        return m_Layers.forward_pass(cast_to_shape<Batch_Size, s_Input_Size>(input_data));
    }

    template <size_t M_Out, size_t N_Out, size_t M_In, size_t N_In>
        requires((M_Out * N_Out == s_Output_Size) && (M_In * N_In == s_Input_Size) && Batch_Size == 1)
    [[nodiscard]] ga_sm::static_matrix<T, M_Out, N_Out> forward_pass(
        ga_sm::static_matrix<T, M_In, N_In> const& input_data
    ) const
    {
        const auto temp = cast_to_shape<1, s_Input_Size>(input_data);
        return cast_to_shape<M_Out, N_Out>(m_Layers.forward_pass(temp));
    }

    [[nodiscard]] value_type forward_pass(value_type input_value) const
        requires((1 == s_Output_Size) && (1 == s_Input_Size) && Batch_Size == 1)
    {
        return m_Layers.forward_pass(ga_sm::static_matrix<value_type, 1, 1>{ input_value })(0, 0);
    }

    // TODO: remove
    // template <size_t Other_Batch_Size, Layer_Signature... Other_Signatures>
    // void init_from_ptr(const static_neural_net<T, Other_Batch_Size, Other_Signatures...>* const src_ptr)
    //     requires(
    //         (sizeof...(Signatures) == sizeof...(Other_Signatures)) &&
    //         std::bool_constant<((Signatures == Other_Signatures) && ...)>::value
    //     )
    // {
    //     std::memcpy(this, src_ptr, sizeof(static_neural_net));
    // }

    template <size_t Other_Batch_Size>
    void init_from_ptr(const static_neural_net<T, Other_Batch_Size, Signatures...>* const src_ptr)
    {
        std::memcpy(this, src_ptr, sizeof(static_neural_net));
    }
};

//--------------------------------------------------------------------------------------//
//  Neural net concept
//--------------------------------------------------------------------------------------//

template <typename T, size_t Batch_Size, Layer_Signature... Signatures>
void nnet_dummy(static_neural_net<T, Batch_Size, Signatures...>)
{
}

template <typename T>
concept static_neural_net_type = requires { nnet_dummy(std::declval<T>()); };

//--------------------------------------------------------------------------------------//
//               Utility
//--------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------//
// Factory functions

template <static_neural_net_type NNet, typename Fn, typename... Args>
    requires std::is_invocable_r_v<typename NNet::value_type, Fn, Args...>
[[nodiscard]] std::unique_ptr<NNet> static_neural_net_factory(Fn&& fn, Args&&... args)
{
    auto ptr = std::make_unique<NNet>();
    ptr->init(fn, std::forward<Args>(args)...);
    return ptr;
}

template <static_neural_net_type NNet>
    requires(std::is_standard_layout_v<NNet> && std::is_trivial_v<NNet>)
[[nodiscard]] bool operator==(const NNet& net1, const NNet& net2)
{
    return std::memcmp(&net1, &net2, sizeof(net1)) == 0;
}

//--------------------------------------------------------------------------------------//
// GA Utility
// neural net x_crossover

template <static_neural_net_type NNet>
std::pair<std::unique_ptr<NNet>, std::unique_ptr<NNet>> net_x_crossover(NNet const& net1, NNet const& net2)
{
    auto ptr_ret_net1 = std::make_unique<NNet>(net1);
    auto ptr_ret_net2 = std::make_unique<NNet>(net2);

    /*
    Si *ptr_net hace una copia local dar� problemas de memoria. Por las pruebas
    que he hecho concluyo que est� optimizado y no se hace. Sino habr�a que
    utilizar esta versi�n que utiliza memcpy y por lo tanto no hace una copia
    local
    */
    // auto ptr_ret_net1 = std::make_unique< static_neural_net<T, Shapes...>>();
    // auto ptr_ret_net2 = std::make_unique< static_neural_net<T, Shapes...>>();
    // ptr_ret_net1->init_from_ptr( ptr_net1 );
    // ptr_ret_net2->init_from_ptr( ptr_net2 );

    // Recursively crossover layers, starting with layer 0
    in_place_net_x_crossover(*ptr_ret_net1.get(), *ptr_ret_net2.get());
    return { std::move(ptr_ret_net1), std::move(ptr_ret_net2) };
}

template <static_layer_type Layer>
inline void in_place_layer_x_crossover(Layer& layer1, Layer& layer2)
{
    in_place_x_crossover(layer1.get_weights_mat(), layer2.get_weights_mat());
    in_place_x_crossover(layer1.get_bias_vector(), layer2.get_bias_vector());
}

template <static_layer_type Layer>
inline void to_target_layer_x_crossover(
    const Layer& in_layer1, const Layer& in_layer2, Layer& out_layer1, Layer& out_layer2
)
{
    to_target_x_crossover(
        in_layer1.get_weights_mat(),
        in_layer2.get_weights_mat(),
        out_layer1.get_weights_mat(),
        out_layer2.get_weights_mat()
    );

    to_target_x_crossover(
        in_layer1.get_bias_vector(),
        in_layer2.get_bias_vector(),
        out_layer1.get_bias_vector(),
        out_layer2.get_bias_vector()
    );
}

template <static_neural_net_type NNet, size_t I = 0>
    requires(I < NNet::s_Layers)
inline void in_place_net_x_crossover(NNet& net1, NNet& net2)
{
    in_place_layer_x_crossover(net1.template layer<I>(), net2.template layer<I>());

    if constexpr (I != NNet::s_Layers - 1)
        in_place_net_x_crossover<NNet, I + 1>(net1, net2);
}

template <static_neural_net_type NNet, size_t I = 0>
    requires(I < NNet::s_Layers)
inline void to_target_net_x_crossover(const NNet& in_net1, const NNet& in_net2, NNet& out_net1, NNet& out_net2)
{
    to_target_layer_x_crossover(
        in_net1.template const_layer<I>(),
        in_net2.template const_layer<I>(),
        out_net1.template layer<I>(),
        out_net2.template layer<I>()
    );

    if constexpr (I != NNet::s_Layers - 1)
        to_target_net_x_crossover<NNet, I + 1>(in_net1, in_net2, out_net1, out_net2);
}

/**
 * \brief Divides NNet layers in two groups (can be thought as encoder and decoder) and crossovers those layer groups in
 * out_net1 and out_net2
 * \note Beware of bad code
 */
template <static_neural_net_type NNet>
    requires(std::is_standard_layout_v<std::remove_reference<NNet>> && std::is_trivial_v<std::remove_reference<NNet>>)
inline void to_target_layer_swap(const NNet& in_net1, const NNet& in_net2, NNet& out_net1, NNet& out_net2)
{
    using value_type = typename NNet::value_type;

    const auto layers         = NNet::s_Layers;
    const auto net_parameters = NNet::parameter_count();

    const auto a = random::randint(0, layers - 1);

    const auto first_half_count  = NNet::parameter_count(0, a);
    const auto second_half_count = NNet::parameter_count(a + 1, layers - 1);
    const auto first_half_size   = first_half_count * sizeof(value_type);
    const auto second_half_size  = second_half_count * sizeof(value_type);

    assert(net_parameters * sizeof(value_type) == first_half_size + second_half_size);
    assert(net_parameters * sizeof(value_type) == sizeof(NNet));

    std::memcpy(&out_net1, &in_net1, first_half_size);
    std::memcpy((value_type*)&out_net1 + first_half_count, (value_type*)&in_net2 + first_half_count, second_half_size);

    std::memcpy(&out_net2, &in_net2, first_half_size);
    std::memcpy((value_type*)&out_net2 + first_half_count, (value_type*)&in_net1 + first_half_count, second_half_size);
}

/**
 * \brief Divides NNet layers in two groups (can be thought as encoder and decoder) and crossovers those layer in place
 * \note Beware of bad code
 */
template <static_neural_net_type NNet>
    requires(std::is_standard_layout_v<std::remove_reference<NNet>> && std::is_trivial_v<std::remove_reference<NNet>>)
inline void in_place_layer_swap([[maybe_unused]] NNet& net1, [[maybe_unused]] NNet& net2)
{
    const auto layers = NNet::s_Layers;
    const auto a      = random::randint(0, layers - 1);

    if (a == 0 || a == layers - 1)
        return;

    const auto first_half_size  = NNet::subnet_size(0, a);
    const auto second_half_size = NNet::subnet_size(a + 1, layers - 1);

    assert(NNet::subnet_size() == first_half_size + second_half_size);

    const auto first = first_half_size < second_half_size;
    const auto start = first ? 0 : first_half_size;
    const auto end   = first ? first_half_size : first_half_size + second_half_size;
    auto       p1    = (char*)&net1;
    auto       p2    = (char*)&net2;

    for (size_t i = start; i != end; ++i)
    {
        helper_functions::pointer_value_swap(p1 + i, p2 + i);
    }
}

template <static_neural_net_type NNet>
std::pair<std::unique_ptr<NNet>, std::unique_ptr<NNet>> layer_swap(NNet const& net1, NNet const& net2)
{
    auto ptr_ret_net1 = std::make_unique<NNet>(net1);
    auto ptr_ret_net2 = std::make_unique<NNet>(net2);

    in_place_layer_swap(*ptr_ret_net1.get(), *ptr_ret_net2.get());
    return { std::move(ptr_ret_net1), std::move(ptr_ret_net2) };
}

//......................................................................................//

//--------------------------------------------------------------------------------------//
// population variability

/**
 * \brief Returns matrix of distances between elements. Diagonal is set to 0s.
 * \param net_ptr_arr Array of pointers to neural nets
 * \return Square matrix containing distances between nets
 */
template <std::floating_point R, size_t N, static_neural_net_type NNet>
    requires(N > 1)
[[nodiscard]] ga_sm::static_matrix<R, N, N> population_variability(
    std::array<std::reference_wrapper<const NNet>, N> const& net_ptr_arr
)
{
    ga_sm::static_matrix<double, N, N> L1_distance_matrix{};
    for (size_t j = 0; j != N - 1; ++j)
    {
        for (size_t i = j + 1; i != N; ++i)
        {
            const auto distance      = L1_net_distance<R>(net_ptr_arr[j].get(), net_ptr_arr[i].get());
            L1_distance_matrix(j, i) = distance;
            L1_distance_matrix(i, j) = distance;
        }
    }
    return L1_distance_matrix;
}

/**
 * \brief Returns the sum of the normalized distance between homologous pairs of matrices between two layers
 * \tparam R
 * \tparam Layer
 * \param layer1
 * \param layer2
 * \return
 */
template <std::floating_point R, static_layer_type Layer1, static_layer_type Layer2>
    requires(
        (Layer1::s_Inputs == Layer2::s_Inputs) && (Layer1::s_Outputs == Layer2::s_Outputs) &&
        (Layer1::s_Activation == Layer2::s_Activation)
    )
[[nodiscard]] R L1_layer_distance(Layer1 const& layer1, Layer2 const& layer2)
{
    return normalized_L1_distance<R>(layer1.get_weights_mat(), layer2.get_weights_mat()) +
        normalized_L1_distance<R>(layer1.get_bias_vector(), layer2.get_bias_vector()) +
        Layer1::activation_function::template L1_distance<R>(
               layer1.get_activation_function(), layer2.get_activation_function()
        );
}

/**
 * \brief Calculates the sum of the normalized distance between homologous pairs of matrices between both nets.
 * Distance between each matrix is normalized by the amount of elements in the matrices.
 * \tparam R Float return trype
 * \tparam NNet1 Net type 1. Must share all layer shapes with Net type 2 as well as the amount of layers.
 * \tparam NNet2 Net type 2. Must share all layer shapes with Net type 1 as well as the amount of layers.
 * \tparam I Implementation detail, should not be used. Used to index layers recursively.
 * \param net1 First net to compare
 * \param net2 Second net to compare
 * \return Sum of the normalized distance between pairs of matrices between both nets.
 */
template <std::floating_point R, static_neural_net_type NNet1, static_neural_net_type NNet2, size_t I = 0>
    requires(NNet1::s_Layers == NNet2::s_Layers)
[[nodiscard]] R L1_net_distance(const NNet1& net1, const NNet2& net2)
{
    const auto current_distance = L1_layer_distance<R>(net1.template const_layer<I>(), net2.template const_layer<I>());
    if constexpr (I == NNet1::s_Layers - 1)
    {
        return current_distance;
    }
    else
    {
        return current_distance + L1_net_distance<R, NNet1, NNet2, I + 1>(net1, net2);
    }
}

//......................................................................................//

} // namespace ga_snn

#endif // !STATIC_NEURAL_NET
