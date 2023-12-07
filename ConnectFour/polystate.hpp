#ifndef TRISTATE_BOARD
#define TRISTATE_BOARD

#include "cx_helper_functions.hpp"
#include <array>
#include <bitset>
#include <cassert>
#include <limits.h>

namespace polystate
{

template <
    std::size_t            N,
    std::unsigned_integral Interface_Type = unsigned int,
    std::unsigned_integral Repr_Type      = std::uint8_t,
    std::size_t            Data_Unit_Bits = 2>
    requires(N > 0) && ((sizeof(Repr_Type) * CHAR_BIT) % Data_Unit_Bits == 0) &&
    ((sizeof(Repr_Type) * CHAR_BIT) >= Data_Unit_Bits) &&
    (Data_Unit_Bits > 0) //&& (sizeof(Repr_Type) >= sizeof(Interface_Type))
class polystate_set
{
public:
    using repr_type = Repr_Type;

    inline static constexpr std::size_t  s_data_units         = N;
    inline static constexpr std::uint8_t s_bits_per_data_unit = Data_Unit_Bits;
    inline static constexpr repr_type    s_max_data_value =
        repr_type{ cx_helper_func::cx_pow(2, s_bits_per_data_unit) - 1 };
    inline static constexpr std::uint8_t s_bits_per_word =
        sizeof(repr_type) * CHAR_BIT;
    inline static constexpr std::uint8_t s_data_units_per_word =
        s_bits_per_word / s_bits_per_data_unit;
    inline static constexpr std::size_t s_word_count =
        N / s_data_units_per_word + (N % s_data_units_per_word == 0 ? 0 : 1);

    using encode_type = std::array<repr_type, s_word_count>;

    struct encode_type_hasher
    {
        using key         = encode_type;
        using result_type = std::size_t;
        using value_type  = typename key::value_type;

        inline static constexpr std::size_t Size = s_word_count;
        inline static constexpr int         data_packet_size =
            sizeof(result_type) / sizeof(value_type);

        [[nodiscard]]
        auto
        operator()(const key& encoded_board) const noexcept -> result_type
        {
            result_type hash{};
            for (std::size_t i = 0; i != Size; ++i)
            {
                auto value = static_cast<std::size_t>(encoded_board[i]);
                auto offs =
                    (i % data_packet_size) * sizeof(value_type) * CHAR_BIT;
                hash ^= value << offs;
            }
            return hash;
        }
    };

    struct encode_type_equal
    {
        using input_type  = encode_type;
        using result_type = bool;

        inline static constexpr std::size_t Size = s_word_count;

        constexpr inline auto operator()(
            const input_type& lhs,
            const input_type& rhs
        ) const noexcept -> bool
        {
            for (size_t i = 0; i != Size; ++i)
            {
                if (lhs[i] != rhs[i])
                    return false;
            }
            return true;
        }
    };

    class reference
    {
        using polystate_type =
            polystate_set<N, Interface_Type, Repr_Type, Data_Unit_Bits>;
        friend polystate_type;

    public:
        constexpr reference& operator=(Interface_Type value) noexcept
        {
            assert(value <= polystate_type::s_max_data_value);
            set(value);
            return *this;
        }

        constexpr reference() noexcept :
            m_ptr_polystate(nullptr),
            m_position(0)
        {
        }

        constexpr reference(
            polystate_type&   polystate,
            const std::size_t pos
        ) noexcept :
            m_ptr_polystate(&polystate),
            m_position(pos)
        {
        }

        constexpr void set(const Interface_Type value) noexcept
        {
            repr_type& selected_word =
                m_ptr_polystate->m_Data[m_position / s_data_units_per_word];
            const repr_type shift_offset =
                (m_position % s_data_units_per_word) * s_bits_per_data_unit;
            const repr_type selected_bits_mask = s_max_data_value
                << shift_offset;

            // Cast value to the repr type in case sizeof(Interface_Type) <
            // sizeof(Repr_Type). The repr_type is guaranteed to be able to hold
            // max word value
            const repr_type repr_type_value = static_cast<repr_type>(value)
                << shift_offset;

            selected_word &= ~selected_bits_mask;
            selected_word ^= repr_type_value;
        }

        constexpr operator Interface_Type() const noexcept
        {
            return const_cast<polystate_type const*>(m_ptr_polystate)
                ->
                operator[](m_position);
        }

    private:
        constexpr reference(reference&&) noexcept      = default;
        constexpr reference(const reference&) noexcept = default;

    public:
        constexpr reference& operator=(reference&&) noexcept      = delete;
        constexpr reference& operator=(const reference&) noexcept = delete;

        ~reference() noexcept = default;

    private:
        polystate_type* m_ptr_polystate;
        std::size_t     m_position;
    };

public:
    [[nodiscard]]
    constexpr auto
    operator[](const std::size_t idx) const noexcept -> repr_type
    {
        assert(idx < N);
        const repr_type shift_offset =
            (idx % s_data_units_per_word) * s_bits_per_data_unit;
        return static_cast<repr_type>(
            (m_Data[idx / s_data_units_per_word] &
             (s_max_data_value << shift_offset)) >>
            shift_offset
        );
    }

    [[nodiscard]]
    constexpr auto
    operator[](const std::size_t idx) noexcept -> reference
    {
        assert(idx < N);
        return reference(*this, idx);
    }

    [[nodiscard]]
    encode_type encode() const
    {
        return m_Data;
    }

    [[nodiscard]]
    static polystate_set decode(const encode_type& encoded_state)
    {
        return polystate_set{ encoded_state };
    }

    encode_type m_Data{};
};

/* ---------------------------------------------------- */

template <
    size_t N,
    typename Return_Type       = int,
    typename Repr_Type         = std::uint8_t,
    std::size_t Data_Unit_Bits = 2>
void polystate_dummy(polystate_set<N, Return_Type, Repr_Type, Data_Unit_Bits>)
{
}

template <typename T>
concept polystate_set_type = requires { polystate_dummy(std::declval<T&>()); };

/* ---------------------------------------------------- */

template <polystate_set_type Polystate>
inline std::ostream& operator<<(std::ostream& os, const Polystate& polystate)
{
    for (size_t i = 0; i != Polystate::s_data_units; ++i)
    {
        os << std::bitset<Polystate::s_bits_per_data_unit>(polystate[i]) << ' ';
    }
    return os;
}

} // namespace polystate

#endif // ! TRISTATE_BOARD
