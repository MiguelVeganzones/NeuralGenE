#ifndef NEURAL_MODEL_DATALOADER
#define NEURAL_MODEL_DATALOADER

#include "c4_board.hpp"
#include "static_matrix.hpp"
#include <array>
#include <cassert>
#include <concepts>
#include <ranges>

namespace encoders
{

struct tristate_encoder
{
    inline static constexpr std::array  s_Encoding_array = { 0.f, 1.f, -1.f };
    inline static constexpr std::size_t s_Values         = s_Encoding_array.size();

    using encode_type = typename decltype(s_Encoding_array)::value_type;

    template <std::integral T>
    [[nodiscard]] static auto encode(T value) -> encode_type
    {
        assert(value >= 0 && value < s_Values);
        return s_Encoding_array[value];
    }
};

template <typename T>
concept tristate_encoder_type = requires { std::is_same_v<T, tristate_encoder>; };

template <typename Encoder>
concept encoder_type = (
    // More valid encoder types could be added here
    tristate_encoder_type<Encoder>
);
} // namespace encoders

namespace data_processor
{
template <encoders::encoder_type Encoder>
struct c4_data_preprocessor
{
    using encoder = Encoder;

    template <c4_board::c4_board_type Input_Type, typename Output_Type>
        requires(std::is_floating_point_v<Output_Type::value_type> && std::is_integral_v<Input_Type::repr_type> &&
                 encoder::s_Values == Input_Type::s_Valid_states)
    [[nodiscard]] static auto process(const Input_Type& board) -> Output_Type
    {
        using board_position = typename Input_Type::board_position;

        auto ret = Output_Type{};
        auto it  = ret.begin();
        for (int j = 0; j != Output_Type::s_Size_y; ++j)
        {
            for (int i = 0; i != Output_Type::s_Size_x; ++i)
            {
                *(it++) = Encoder::encode(board.at(board_position{ j, i }));
            }
        }
        assert(it == ret.end());
        return ret;
    }
};

// iterable concept
template <typename T>
concept iterable_type = requires(T& v) {
                            v.begin();
                            v.end();
                            typename T::value_type;
                            v.size();
                        };

struct iterable_indexer
{

    inline static int index = -1;

    template <iterable_type Input_Type, typename Output_Type>
        requires std::is_convertible_v<typename Input_Type::value_type, Output_Type>
    [[nodiscard]] static auto process(const Input_Type& input_range) -> Output_Type
    {
        assert(index >= 0 && index < input_range.size());
        return static_cast<Output_Type>(*(input_range.begin() + index));
    }
};

struct scalar_converter
{
    inline static constexpr int index = 0;

    template <typename Input_Type, typename Output_Type>
    [[nodiscard]] static auto process(Input_Type input_value) -> Output_Type
    {
        return static_cast<Output_Type>(input_value);
    }

    template <iterable_type Input_Type, typename Output_Type>
    [[nodiscard]] static auto process(Input_Type input_range) -> Output_Type
    {
        return static_cast<Output_Type>(*input_range.begin());
    }

    template <typename Input_Type, iterable_type Output_Type>
    [[nodiscard]] static auto process(Input_Type input_value) -> Output_Type
    {
        Output_Type ret{};
        *ret.begin() = static_cast<Output_Type::value_type>(input_value);
        return ret;
    }
};

// -------------------------------------------------------------------------- //
// Datalaoder concepts

template <typename Board>
void c4_data_processor_dummy(c4_data_preprocessor<Board>)
{
}

template <typename T>
concept c4_data_processor_type = requires { c4_data_processor_dummy(std::declval<T>()); };

template <typename T>
concept iterable_indexer_type = std::is_same_v<T, iterable_indexer>;

template <typename T>
concept scalar_converter_type = std::is_same_v<T, scalar_converter>;

template <typename T>
concept data_processor_type = (c4_data_processor_type<T> || iterable_indexer_type<T> || scalar_converter_type<T>);

} // namespace data_processor

#endif // !NEURAL_MODEL_DATALOADER