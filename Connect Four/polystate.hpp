#pragma once

#ifndef TRISTATE_BOARD
#define TRISTATE_BOARD

#include <bitset>

#include "cx_helper_functions.h"

#include <cassert>

namespace polystate
{


template <size_t                 N,
          std::unsigned_integral Interface_Type = unsigned int,
          std::unsigned_integral Repr_Type      = std::uint8_t,
          size_t                 Data_Unit_Bits = 2>
    requires(N > 0) && ((sizeof(Repr_Type) * CHAR_BIT) % Data_Unit_Bits == 0) &&
    ((sizeof(Repr_Type) * CHAR_BIT) >= Data_Unit_Bits) &&
    (Data_Unit_Bits > 0) //&& (sizeof(Repr_Type) >= sizeof(Interface_Type))
class polystate_set
{
public:
    using repr_type = Repr_Type;

    inline static constexpr size_t    s_data_units         = N;
    inline static constexpr size_t    s_bits_per_data_unit = Data_Unit_Bits;
    inline static constexpr repr_type s_max_data_value =
        repr_type{ cx_helper_func::cx_pow(2, s_bits_per_data_unit) - 1 };
    inline static constexpr size_t s_bits_per_word       = sizeof(repr_type) * CHAR_BIT;
    inline static constexpr size_t s_data_units_per_word = s_bits_per_word / s_bits_per_data_unit;
    inline static constexpr size_t s_word_count = N / s_data_units_per_word + (N % s_data_units_per_word == 0 ? 0 : 1);

    using encode_type = std::array<repr_type, s_word_count>;

    class reference
    {
        using polystate = polystate_set<N, Interface_Type, Repr_Type, Data_Unit_Bits>;
        friend polystate;

    public:
        constexpr reference& operator=(Interface_Type value) noexcept
        {
            assert(value <= polystate::s_max_data_value);
            set(value);
            return *this;
        }

        constexpr reference() noexcept : m_ptr_polystate(nullptr), m_position(0)
        {
        }

        constexpr reference(polystate& polystate, const size_t pos) noexcept :
            m_ptr_polystate(&polystate), m_position(pos)
        {
        }

        constexpr void set(const Interface_Type value) noexcept
        {
            auto&      selected_word      = m_ptr_polystate->m_Data[m_position / s_data_units_per_word];
            const auto shift_offset       = (m_position % s_data_units_per_word) * s_bits_per_data_unit;
            const auto selected_bits_mask = s_max_data_value << shift_offset;

            // Cast value to the repr type in case sizeof(Interface_Type) < sizeof(Repr_Type). The repr_type is
            // guaranteed to be able to hold max word value
            const auto repr_type_value = static_cast<repr_type>(value) << shift_offset;


            selected_word &= ~selected_bits_mask;
            selected_word ^= repr_type_value;
        }

        constexpr operator Interface_Type() const noexcept
        {
            return const_cast<polystate const*>(m_ptr_polystate)->operator[](m_position);
        }

    private:
        constexpr reference(reference&&) noexcept      = default;
        constexpr reference(const reference&) noexcept = default;

    public:
        constexpr reference& operator=(reference&&) noexcept      = delete;
        constexpr reference& operator=(const reference&) noexcept = delete;

        ~reference() noexcept = default;

    private:
        polystate* m_ptr_polystate;
        size_t     m_position;
    };

public:
    [[nodiscard]] constexpr Interface_Type operator[](const size_t idx) const noexcept
    {
        assert(idx < N);
        const auto shift_offset = ((idx % s_data_units_per_word) * s_bits_per_data_unit);
        return (m_Data[idx / s_data_units_per_word] & (s_max_data_value << shift_offset)) >> shift_offset;
    }

    [[nodiscard]] constexpr reference operator[](const size_t idx) noexcept
    {
        assert(idx < N);
        return reference(*this, idx);
    }

    [[nodiscard]] encode_type encode() const
    {
        return m_Data;
    }

    static [[nodiscard]] polystate_set decode(const encode_type& encoded_state)
    {
        return polystate_set{ encoded_state };
    }

    encode_type m_Data{};
};

/* ---------------------------------------------------- */

template <size_t N, typename Return_Type = int, typename Repr_Type = std::uint8_t, size_t Data_Unit_Bits = 2>
void polystate_dummy(polystate_set<N, Return_Type, Repr_Type, Data_Unit_Bits>)
{
}

template <typename T>
concept polystate_set_type = requires { polystate_dummy(std::declval<T>()); };

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