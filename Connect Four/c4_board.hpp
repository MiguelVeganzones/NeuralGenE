#ifndef CONNECT_FOUR_BOARD
#define CONNECT_FOUR_BOARD

#include "Random.h"
#include "game_boardd.hpp"
#include <array>
#include <cassert>
#include <ranges>

namespace c4_board
{
template <size_t Size_Y, size_t Size_X, size_t Target_Count, std::uint8_t Player_Count>
requires(Target_Count > 1) && (Size_X >= Target_Count) && (Size_Y >= Target_Count) && (Player_Count > 1) class board
{
public:
    using board_position   = typename game_board::board_2D_position;
    using player_repr_type = std::uint8_t;

private:
    inline static constexpr int              s_Size_y          = Size_Y;
    inline static constexpr int              s_Size_x          = Size_X;
    inline static constexpr int              s_Max_moves_count = s_Size_x;
    inline static constexpr int              s_Target_count    = Target_Count;
    inline static constexpr player_repr_type s_Player_count    = Player_Count;
    inline static constexpr player_repr_type s_Valid_states    = Player_Count + 1;

public:
    using game_board_type         = game_board::board_2D<s_Size_y, s_Size_x, s_Valid_states, player_repr_type>;
    using repr_type               = typename game_board_type::repr_type;
    using move_type               = board_position;
    using valid_moves_container   = std::array<move_type, s_Size_x>;
    using encode_type             = typename game_board_type::encode_type;
    using valid_actions_container = valid_moves_container;
    using action_type             = typename valid_actions_container::value_type;

public:
    board(const game_board_type& board_type) : m_Board_state{ board_type }
    {
        m_Valid_moves = calculate_valid_moves();
    }

    board() = default;

    [[nodiscard]] auto get_valid_actions() const -> const valid_actions_container&
    {
        return get_valid_moves();
    }

    auto perform_action(const action_type pos) -> void
    {
        make_move(pos);
    }

    [[nodiscard]] auto any_actions_left() const -> bool
    {
        return any_moves_left();
    }

    void make_move(const move_type pos, const player_repr_type player)
    {
        assert(pos.y < s_Size_y);
        assert(pos.x < s_Size_x);
        assert(player <= s_Player_count);
        assert(m_Board_state.is_empty(pos));

        // update board state
        m_Board_state.set_position(pos, player_to_repr_value(player));
        // update valid moves
        --m_Valid_moves[pos.x].y;
    }

    auto make_move(const move_type pos) -> void
    {
        make_move(pos, m_Current_player);
        set_next_player();
    }

    auto undo_move(const move_type pos) -> void
    {
        assert(pos.y < s_Size_y);
        assert(pos.x < s_Size_x);
        assert(!m_Board_state.is_empty(pos));
        assert((pos.y > 0) ? m_Board_state.is_empty(game_board_type::up(pos)) : true);

        // update board state
        m_Board_state.set_position(pos, m_Board_state.s_Empty_state_value);
        // update valid moves
        ++m_Valid_moves[pos.x].y;
        set_previous_player();
    }

    [[nodiscard]] auto select_random_move() const -> move_type
    {
        std::array<int, s_Max_moves_count> valid_idx{};

        int valid_moves = 0;
        // Number of valid moves
        for (int i = 0; i != s_Max_moves_count; ++i)
        {
            if (m_Valid_moves[i])
            {
                valid_idx[valid_moves++] = i;
            }
        }

        return m_Valid_moves[valid_idx[random::randint(0, valid_moves - 1)]];
    }

    [[nodiscard]] auto at(const board_position pos) const -> repr_type
    {
        return m_Board_state.at(pos);
    }

    [[nodiscard]] auto board_state() const -> game_board_type const&
    {
        return m_Board_state;
    }

    [[nodiscard]] const valid_moves_container& get_valid_moves() const
    {
        return m_Valid_moves;
    }

    [[nodiscard]] bool is_valid_move(const move_type pos) const
    {
        return m_Board_state.is_empty(pos) &&
            (pos.y == s_Size_y - 1 ? true : !m_Board_state.is_empty(game_board_type::down(pos)));
    }

    [[nodiscard]] bool any_moves_left() const
    {
        return std::ranges::any_of(m_Valid_moves, [](auto e) { return static_cast<bool>(e); });
    }

    [[nodiscard]] bool winning_move(const move_type pos) const
    {
        const auto player = m_Board_state.at(pos);

        return check_left_diagonal(pos, player) || check_right_diagonal(pos, player) || check_horizontal(pos, player) ||
            check_vertical(pos, player);
    }

    [[nodiscard]] bool check_left_diagonal(const board_position pos, int player = -1) const
    {
        ++leftd_count;

        if (player < 0)
            player = m_Board_state.at(pos);
        else
            assert(m_Board_state.at(pos) == player);

        const int x = pos.x;
        const int y = pos.y;

        int count = 1;

        for (int i = x - 1, j = y - 1; i >= 0 && j >= 0; --i, --j)
        {
            if (m_Board_state.at({ j, i }) == player)
                ++count;
            else
                break;
        }

        for (int i = x + 1, j = y + 1; i < s_Size_x && j < s_Size_y; ++i, ++j)
        {
            if (m_Board_state.at({ j, i }) == player)
                ++count;
            else
                break;
        }

        return count >= s_Target_count;
    }

    [[nodiscard]] bool check_right_diagonal(const board_position pos, int player = -1) const
    {
        ++rightd_count;

        if (player < 0)
            player = m_Board_state.at(pos);
        else
            assert(m_Board_state.at(pos) == player);

        const int x = pos.x;
        const int y = pos.y;

        int count = 1;

        for (int i = x - 1, j = y + 1; i >= 0 && j < s_Size_y; --i, ++j)
        {
            if (m_Board_state.at({ j, i }) == player)
                ++count;
            else
                break;
        }

        for (int i = x + 1, j = y - 1; i < s_Size_x && j >= 0; ++i, --j)
        {
            if (m_Board_state.at({ j, i }) == player)
                ++count;
            else
                break;
        }

        return count >= s_Target_count;
    }

    [[nodiscard]] bool check_horizontal(const board_position pos, int player = -1) const
    {
        ++horizontal_count;

        if (player < 0)
            player = m_Board_state.at(pos);
        else
            assert(m_Board_state.at(pos) == player);

        const auto x = pos.x;
        const auto y = pos.y;

        int count = 1;

        for (int i = x - 1, j = y; i >= 0; --i)
        {
            if (m_Board_state.at({ j, i }) == player)
                ++count;
            else
                break;
        }

        for (int i = x + 1, j = y; i < s_Size_x; ++i)
        {
            if (m_Board_state.at({ j, i }) == player)
                ++count;
            else
                break;
        }

        return count >= s_Target_count;
    }

    [[nodiscard]] bool check_vertical(const board_position pos, int player = -1) const
    {
        ++vertical_count;

        if (player < 0)
            player = m_Board_state.at(pos);
        else
            assert(m_Board_state.at(pos) == player);

        const auto x = pos.x;
        const auto y = pos.y;

        int count = 1;

        // Check vertical
        if (s_Size_y - y < s_Target_count)
            return false;

        for (int i = x, j = y + 1, iter = 0; iter != s_Target_count - 1; ++j, ++iter)
        {
            if (m_Board_state.at({ j, i }) == player)
                ++count;
            else
                break;
        }

        return count >= s_Target_count;
    }

    [[nodiscard]] encode_type encode() const
    {
        return m_Board_state.encode();
    }

    [[nodiscard]] static board decode(const encode_type& encoded_board)
    {
        board b{};
        b.m_Board_state = game_board_type::decode(encoded_board);
        b.m_Valid_moves = b.calculate_valid_moves();
        b.calculate_current_player();
        return b;
    }

    player_repr_type current_player() const
    {
        return m_Current_player;
    }

    player_repr_type previous_player() const
    {
        return m_Current_player == 0 ? s_Player_count - 1 : m_Current_player - 1;
    }

private:
    static inline player_repr_type player_to_repr_value(const player_repr_type player)
    {
        assert(player < s_Valid_states - 1);
        return player + 1;
    }

    static inline player_repr_type repr_value_to_player(const player_repr_type board_value)
    {
        assert(board_value < s_Valid_states);
        assert(board_value > 0);
        return board_value - 1;
    }

    static constexpr inline valid_moves_container init_valid_moves()
    {
        valid_moves_container initial_moves{};
        for (int i = 0; i != s_Size_x; ++i)
        {
            initial_moves[i] = board_position{ s_Size_y - 1, i };
        }
        return initial_moves;
    }

    [[nodiscard]] valid_moves_container calculate_valid_moves() const
    {
        valid_moves_container moves = init_valid_moves();
        for (auto& move : moves)
        {
            while (move)
            {
                if (is_valid_move(move))
                    break;
                --move.y;
            }
        }
        return moves;
    }

    void set_next_player()
    {
        ++m_Current_player %= s_Player_count;
    }

    void set_previous_player()
    {
        m_Current_player = previous_player();
    }

    void calculate_current_player()
    {
        size_t moves_count = 0;
        for (auto& e : m_Valid_moves)
        {
            moves_count += s_Size_y - e.y - 1;
        }
        m_Current_player = moves_count % s_Player_count;
    }

private:
    game_board_type       m_Board_state{};
    valid_moves_container m_Valid_moves    = init_valid_moves();
    player_repr_type      m_Current_player = 0;

public:
    // FIXME: Remove
    mutable int leftd_count      = 0;
    mutable int rightd_count     = 0;
    mutable int vertical_count   = 0;
    mutable int horizontal_count = 0;
};

//--------------------------------------------------------------------------------------//
//  C4 board concept
//--------------------------------------------------------------------------------------//

template <size_t Size_Y, size_t Size_X, size_t Target_Count, std::uint8_t Player_Count>
void c4_board_dummy(board<Size_Y, Size_X, Target_Count, Player_Count>)
{
}

template <typename T>
concept c4_board_type = requires
{
    c4_board_dummy(std::declval<T>());
};

} // namespace c4_board

#endif // ! CONNECT_FOUR_BOARD
