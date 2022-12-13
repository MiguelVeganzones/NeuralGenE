#pragma once

#ifndef STATIC_NEURAL_NET
#define STATIC_NEURAL_NET

#include <array>
#include <concepts>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "Log.h"
#include "activation_functions.hpp"
#include "static_matrix.hpp"

namespace ga_snn
{

/*
  Each element of the matrix might be:
  Replaced by a value given by fn(args...) with a probability p1 : e =
fn(args...)
  - or -
  Modified by a normal distribuion given by [avg, Stddev] and fn(args...) with
a probability p2 : e
+= N(abg, Stddev)*fn(args...)
>>> void mutate(float Avg, float Stddev, float p1, float p2)
*/
struct mutate_params
{
    float p1;
    float p2;
    float Avg;
    float Stddev;
};

struct Layer_Signature
{
    size_t                                   Size;
    matrix_activation_functions::Identifiers Activation;

    constexpr bool operator==(const Layer_Signature&) const = default;
};

struct Layer_Structure
{
    size_t                                   Inputs;
    size_t                                   Outputs;
    matrix_activation_functions::Identifiers Activation;

    constexpr bool operator==(const Layer_Structure&) const = default;
};

//--------------------------------------------------------------------------------------//

template <typename T, size_t Batch_Size, Layer_Structure Structure>
    requires(std::is_floating_point_v<T> && Batch_Size > 0)
class layer
{
private:
    static constexpr size_t s_Inputs  = Structure.Inputs;
    static constexpr size_t s_Outputs = Structure.Outputs;

public:
    using weights_shape       = ga_sm::static_matrix<T, s_Inputs, s_Outputs>;
    using output_vector_shape = ga_sm::static_matrix<T, Batch_Size, s_Outputs>;
    using input_vector_shape  = ga_sm::static_matrix<T, Batch_Size, s_Inputs>;
    using bias_vector_shape   = ga_sm::static_matrix<T, 1, s_Outputs>;

private:
    inline static constexpr auto s_Activation_Function =
        matrix_activation_functions::choose_func<output_vector_shape, Structure.Activation>();
    // inline static std::function<T(T)>
    //     s_Activation_Function = activation_functions::choose_func<T,
    //     Structure.Activation>();

private:
    weights_shape     m_weights_mat;
    bias_vector_shape m_bias_vector;

public:
    // each neuron produces a column vector that splits into the input of the
    // neurons in the next layer
    [[nodiscard]] constexpr output_vector_shape forward_pass(input_vector_shape const& Input) const
    {
        // return multiply_add_activate(m_weights_mat, Input, m_offset_vector,
        // s_Activation_Function);

        return s_Activation_Function(matrix_vec_add(matrix_mul(Input, m_weights_mat), m_bias_vector));
    }

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<T, Fn, Args...>
    constexpr void init(Fn fn, Args... args)
    {
        m_weights_mat.fill(fn, args...);
        m_bias_vector.fill(fn, args...);
    }

    template <typename Mutate_params, typename Fn, typename... Args>
    constexpr void mutate(const Mutate_params& params, Fn fn, Args... args)
    {
        m_weights_mat.mutate_replace_add(params.p1, params.p2, params.Avg, params.Stddev, fn, args...);
        m_bias_vector.mutate_replace_add(params.p1, params.p2, params.Avg, params.Stddev, fn, args...);
    }

    void store(std::ofstream& out) const
    {
        m_weights_mat.store(out);
        m_bias_vector.store(out);
    }

    void load(std::ifstream& in)
    {
        m_weights_mat.load(in);
        m_bias_vector.load(in);
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

    [[nodiscard]] static inline constexpr size_t get_inputs()
    {
        return s_Inputs;
    }

    [[nodiscard]] static inline constexpr size_t get_outputs()
    {
        return s_Outputs;
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
    os << layer.get_weights_mat() << layer.get_bias_vector() << "\n";
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

    layer<T, Batch_Size, Layer_Structure{ Inputs, Current_Signature.Size, Current_Signature.Activation }>
        m_Data; // one data member for this layer

    //------ MEMBER FUNCTIONS -----//
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
            static_assert(false && Idx, "requested layer index exceeds number of layers");
        }
    }

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<T, Fn, Args...>
    void init(Fn fn, Args... args)
    {
        m_Data.init(fn, args...);
    }

    template <typename Mutate_params, typename Fn, typename... Args>
    void mutate(const Mutate_params& params, Fn fn, Args... args)
    {
        m_Data.mutate(params, fn, args...);
    }

    void print() const
    {
        std::cout << m_Data << "\n";
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
};

// general specialization
template <typename T,
          size_t          Inputs,
          size_t          Batch_Size,
          Layer_Signature Current_Signature,
          Layer_Signature... Signatures>
struct layer_unroll<T, Inputs, Batch_Size, Current_Signature, Signatures...>
{
    static constexpr size_t s_Inputs{ Inputs };
    static constexpr size_t s_Outputs{ Current_Signature.Size };

    layer<T, Batch_Size, Layer_Structure{ s_Inputs, s_Outputs, Current_Signature.Activation }>
        m_Data; // one data member for this layer
    layer_unroll<T, Current_Signature.Size, Batch_Size, Signatures...>
        m_Next; // another layer_unroll member for the rest

    //------ MEMBER FUNCTIONS -----//
    // getter so that we can actually get a specific layer by index
    template <size_t Idx>
    constexpr auto& get()
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
    [[nodiscard]] constexpr auto const& const_get() const
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
    void init(Fn fn, Args... args)
    {
        m_Data.init(fn, args...);
        m_Next.init(fn, args...);
    }

    template <typename Mutate_params, typename Fn, typename... Args>
    void mutate(const Mutate_params& params, Fn fn, Args... args)
    {
        m_Data.mutate(params, fn, args...);
        m_Next.mutate(params, fn, args...);
    }

    void print() const
    {
        std::cout << m_Data << "\n";
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
};

//--------------------------------------------------------------------------------------//

template <typename T, size_t Batch_Size, Layer_Signature... Signatures>
    requires((sizeof...(Signatures) >= 3) && std::is_floating_point_v<T> and Batch_Size > 0)
class static_neural_net
{
public:
    static constexpr size_t                                s_Layers{ sizeof...(Signatures) };
    static constexpr std::array<Layer_Signature, s_Layers> s_Signatures{ Signatures... };
    static constexpr size_t                                s_Output_Size = s_Signatures[s_Layers - 1].Size;
    static constexpr size_t                                s_Input_Size  = s_Signatures[0].Size;

private:
    layer_unroll<T, s_Input_Size, Batch_Size, Signatures...> m_Layers;

public:
    using value_type  = T;
    using input_type  = ga_sm::static_matrix<T, Batch_Size, s_Input_Size>;
    using output_type = ga_sm::static_matrix<T, Batch_Size, s_Output_Size>;

public:
    constexpr static size_t parameter_count()
    {
        size_t p = (s_Input_Size + 1) * s_Input_Size;
        for (size_t i = 1; i < s_Layers; ++i)
        {
            p += (s_Signatures[i - 1].Size + 1) * s_Signatures[i].Size;
        }
        return p;
    }

    template <size_t Batch_Size_Other>
    void init_from_ptr(const static_neural_net<T,
                                               Batch_Size_Other,
                                               Signatures...>* const ptr_other)
        requires(std::is_standard_layout_v<static_neural_net> &&
                 std::is_trivial_v<static_neural_net> // creo que no hacen falta
                                                      // las dos pero bueno
        )
    {
        std::memcpy(this, ptr_other, sizeof(static_neural_net));
    }

    template <std::size_t Idx>
    auto& layer()
    {
        return m_Layers.template get<Idx>();
    } // you arent a real template programmer if you dont use .template  /s

    template <std::size_t Idx>
    [[nodiscard]] auto const& const_layer() const
    {
        return m_Layers.template const_get<Idx>();
    }

    template <typename Fn, typename... Args>
        requires std::is_invocable_r_v<T, Fn, Args...>
    void init(Fn fn, Args... args)
    {
        m_Layers.init(fn, args...);
    }

    template <typename Mutate_params, typename Fn, typename... Args>
    void mutate(const Mutate_params& params, Fn fn, Args... args)
    {
        m_Layers.mutate(params, fn, args...);
    }

    void print_layers() const
    {
        std::ios_base::sync_with_stdio(false);
        m_Layers.print();
        std::ios_base::sync_with_stdio(true);
    }

    void print_net() const
    {
        std::cout << "#############################################################"
                     "#############\n";
        print_layers();
        std::cout << "Net has " << parameter_count() << " parameters\n";
        std::cout << "Net address " << this << "\n";
        std::cout << "-------------------------------------------------------------"
                     "-------------\n";
    }

    void store(const std::string& filename) const
    {
        std::ofstream out(filename);
        if (!out.is_open())
        {
            const auto message = "Could not create file: " + filename + "\n";
            std::cout << message;
            log::add(message);
            exit(EXIT_FAILURE);
        }
        // store net shapes
        for (const auto& layer_signature : s_Signatures)
        {
            out << layer_signature.Size << " ";
        }
        out << "\n\n";
        m_Layers.store(out); // Store layers recursively
        out.close();
        if (!out)
        {
            const auto message = "Could not close file: " + filename + "\n";
            std::cout << message;
            log::add(message);
            exit(EXIT_FAILURE);
        }
    }

    // Overrides current net with one read from "filename"
    // Shapes of both nets must be the same
    void load(const std::string& filename)
    {
        std::ifstream in(filename);
        if (!in.is_open())
        {
            const auto message = "Could not open file: " + filename + "\n";
            std::cout << message;
            log::add(message);
            exit(EXIT_FAILURE);
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
                exit(EXIT_FAILURE);
            }
        }
        m_Layers.load(in); // load layers recursively
        in.close();
        if (!in)
        {
            const auto message = "Could not close file after reading: " + filename + "\n";
            std::cout << message;
            log::add(message);
        }
    }

    [[nodiscard]] output_type batch_forward_pass(input_type const& input_data) const
    {
        return m_Layers.forward_pass(input_data);
    }

    template <size_t M_Out, size_t N_Out, size_t M_In, size_t N_In>
        requires((M_Out * N_Out == s_Output_Size) && (M_In * N_In == s_Input_Size) && Batch_Size == 1)
    [[nodiscard]] ga_sm::static_matrix<T, M_Out, N_Out> forward_pass(
        ga_sm::static_matrix<T, M_In, N_In> const& input_data) const
    {
        const auto temp = cast_to_shape<1, s_Input_Size>(input_data);
        return cast_to_shape<M_Out, N_Out>(m_Layers.forward_pass(temp));
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
    requires std::is_invocable_v<Fn, Args...>
[[nodiscard]] std::unique_ptr<NNet> static_neural_net_factory(Fn fn, Args... args)
{
    using T = typename NNet::value_type;
    static_assert(std::is_invocable_r_v<T, Fn, Args...>);

    auto ptr = std::make_unique<NNet>();
    ptr->init(fn, args...);
    return std::move(ptr);
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
    Si *ptr_net hace una copia local dará problemas de memoria. Por las pruebas
    que he hecho concluyo que está optimizado y no se hace. Sino habría que
    utilizar esta versión que utiliza memcpy y por lo tanto no hace una copia
    local
    */
    // auto ptr_ret_net1 = std::make_unique< static_neural_net<T, Shapes...>>();
    // auto ptr_ret_net2 = std::make_unique< static_neural_net<T, Shapes...>>();
    // ptr_ret_net1->init_from_ptr( ptr_net1 );
    // ptr_ret_net2->init_from_ptr( ptr_net2 );

    // Recursively crossover layers, starting with layer 0
    in_place_net_x_crossover(*ptr_ret_net1.get(), *ptr_ret_net2.get());
    return std::make_pair(std::move(ptr_ret_net1), std::move(ptr_ret_net2));
}

template <static_layer_type Layer>
inline void in_place_layer_x_crossover(Layer& layer1, Layer& layer2)
{
    in_place_x_crossover(layer1.get_weights_mat(), layer2.get_weights_mat());

    in_place_x_crossover(layer1.get_bias_vector(), layer2.get_bias_vector());
}

template <static_layer_type Layer>
inline void to_target_layer_x_crossover(const Layer& in_layer1,
                                        const Layer& in_layer2,
                                        Layer&       out_layer1,
                                        Layer&       out_layer2)
{
    to_target_x_crossover(in_layer1.get_weights_mat(),
                          in_layer2.get_weights_mat(),
                          out_layer1.get_weights_mat(),
                          out_layer2.get_weights_mat());

    to_target_x_crossover(in_layer1.get_bias_vector(),
                          in_layer2.get_bias_vector(),
                          out_layer1.get_bias_vector(),
                          out_layer2.get_bias_vector());
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
    to_target_layer_x_crossover(in_net1.template const_layer<I>(),
                                in_net2.template const_layer<I>(),
                                out_net1.template layer<I>(),
                                out_net2.template layer<I>());

    if constexpr (I != NNet::s_Layers - 1)
        to_target_net_x_crossover<NNet, I + 1>(in_net1, in_net2, out_net1, out_net2);
}
//......................................................................................//

//--------------------------------------------------------------------------------------//
// population variability

// Returns matrix of distances between elements
// Diagonal is set to 0
template <size_t N, static_neural_net_type NNet>
    requires(N > 1)
[[nodiscard]] ga_sm::static_matrix<double, N, N> population_variability(std::array<NNet*, N> const& net_ptr_arr)
{
    ga_sm::static_matrix<double, N, N> L11_distance_matrix{};
    for (size_t j = 0; j < N; ++j)
    {
        for (size_t i = j + 1; i < N; ++i)
        {
            double distance           = L11_net_distance(net_ptr_arr[j], net_ptr_arr[i]);
            L11_distance_matrix(j, i) = distance;
            L11_distance_matrix(i, j) = distance;
        }
    }
    return L11_distance_matrix;
}

template <static_layer_type Layer>
[[nodiscard]] double L11_layer_distance(Layer const& layer1, Layer const& layer2)
{
    return normalized_L1_distance(layer1.get_weights_mat(), layer2.get_weights_mat()) +
        normalized_L1_distance(layer1.get_bias_vector(), layer2.get_bias_vector());
}

/// <summary>
/// Returns the L11 distance between two nets.
/// </summary>
/// <template param name="I"> Index of the first layer to compare. Should be set
/// to zero to compare the whole net. </template param> <typeparam
/// name="T"></typeparam> <param name="ptr_net1"></param> <param
/// name="ptr_net2"></param> <returns></returns>
template <static_neural_net_type NNet, size_t I = 0>
[[nodiscard]] double L11_net_distance(const NNet* const ptr_net1, const NNet* const ptr_net2)
{
    const auto current_distance =
        L11_layer_distance(ptr_net1->template const_layer<I>(), ptr_net2->template const_layer<I>());
    if constexpr (I == NNet::s_Layers - 1)
    {
        return current_distance;
    }
    else
    {
        return current_distance + L11_net_distance<NNet, I + 1>(ptr_net1, ptr_net2);
    }
}

//......................................................................................//

} // namespace ga_snn

#endif // !STATIC_NEURAL_NET
