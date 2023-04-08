#ifndef GAME_BOARD
#define GAME_BOARD
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <limits.h>
#include <limits>
#include <type_traits>

#include "cx_helper_functions.h"
#include "polystate.hpp"

#ifdef max
#undef max
#endif

namespace game_board
{

struct board_2D_position
{
    int y = -1;
    int x = -1;

    operator bool() const
    {
        return y >= 0 && x >= 0;
    }
};

inline std::ostream& operator<<(std::ostream& os, const board_2D_position& pos)
{
    os << '(' << pos.x << ", " << pos.y << ')';
    return os;
}

template <
    size_t                 Size_Y,
    size_t                 Size_X,
    size_t                 Valid_States,
    std::unsigned_integral Repr_Type,
    Repr_Type              Empty_State = Repr_Type{}>
    requires(cx_helper_func::cx_pow(2, sizeof(Repr_Type) * CHAR_BIT) >= Valid_States) &&
    (Size_X < std::numeric_limits<int>::max()) && (Size_Y < std::numeric_limits<int>::max())
class board_2D
{
public:
    using repr_type      = Repr_Type;
    using interface_type = Repr_Type;

    inline static constexpr repr_type  s_Empty_state_value = Empty_State;
    inline static constexpr size_t     s_Bits_per_state    = cx_helper_func::cx_pow2_bits_required(Valid_States);
    inline static constexpr size_t     s_Size              = Size_X * Size_Y;
    inline static constexpr size_t     s_Size_x            = Size_X;
    inline static constexpr size_t     s_Size_y            = Size_Y;
    inline static constexpr size_t     s_Valid_states      = Valid_States;
    inline static constexpr std::array s_Char_state_repr   = { ' ', 'x', 'o', '1', '2', '3', '4', '5', '6',
                                                               '7', '8', '9', '0', '#', '+', '*', '?', '&' };

    using board_state_type = polystate::polystate_set<s_Size, interface_type, repr_type, s_Bits_per_state>;
    using encode_type      = typename board_state_type::encode_type;
    using hash_function    = board_state_type::hash_function;

    auto reset(const repr_type value = s_Empty_state_value) -> void
    {
        assert(value < s_Valid_states);
        for (size_t i = 0; i != s_Size; ++i)
        {
            m_State[i] = value;
        }
    }

    auto set_position(const board_2D_position pos, const interface_type value) -> void
    {
        assert(value < s_Valid_states);
        m_State[position(pos)] = value;
    }

    [[nodiscard]] constexpr auto at(const board_2D_position pos) const -> repr_type
    {
        return m_State[position(pos)];
    }

    [[nodiscard]] constexpr bool is_empty(const board_2D_position pos) const
    {
        return at(pos) == s_Empty_state_value;
    }

    inline static constexpr size_t position(const board_2D_position pos)
    {
        assert(pos.x >= 0);
        assert(pos.y >= 0);

        return pos.y * s_Size_x + pos.x;
    }

    inline static constexpr char pretty_repr(const repr_type value)
    {
        return s_Char_state_repr[value];
    }

    [[nodiscard]] encode_type encode() const
    {
        return m_State.encode();
    }

    static board_2D decode(const encode_type& encoded_board2D)
    {
        return board_2D{ board_state_type::decode(encoded_board2D) };
    }

    [[nodiscard]] static board_2D_position up(const board_2D_position pos)
    {
        return board_2D_position{ pos.y - 1, pos.x };
    }

    [[nodiscard]] static board_2D_position down(const board_2D_position pos)
    {
        return board_2D_position{ pos.y + 1, pos.x };
    }

public:
    board_state_type m_State;
};

template <size_t Size_Y, size_t Size_X, size_t Valid_States, std::unsigned_integral Repr_Type, Repr_Type Empty_State>
void game_board_2D_dummy(board_2D<Size_Y, Size_X, Valid_States, Repr_Type, Empty_State>)
{
}

template <typename T>
concept game_board_type = requires { game_board_2D_dummy(std::declval<T>()); };

/* ---------------------------------------------------- */

template <game_board_type Game_Board>
inline std::ostream& operator<<(std::ostream& os, const Game_Board& board)
{
    for (int j = 0; j != Game_Board::s_Size_y; ++j)
    {
        for (int i = 0; i != Game_Board::s_Size_x; ++i)
        {
            os << Game_Board::pretty_repr(board.at({ j, i })) << ", ";
        }
        os << '\n';
    }
    return os;
}

/* ---------------------------------------------------- */

} // namespace game_board


#endif // ! GAME_BOARD
