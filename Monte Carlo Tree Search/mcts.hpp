#ifndef MONTE_CARLO_TREE_SEARCH
#define MONTE_CARLO_TREE_SEARCH

#include "Random.h"

#include <cassert>
#include <list>
#include <map>
#include <vector>


namespace mcts
{
template <class Game_Board>
class mtcs_search_engine;

struct mtcs_results
{
    using points_type       = float;
    using sample_count_type = int;
    points_type       points{};
    sample_count_type samples{};

    constexpr void add_trial_score(const points_type trial_score) noexcept
    {
        points += trial_score;
        ++samples;
    }
};

template <class Game_Encode_Type>
struct mtcs_flat_tree_node
{
    using flat_tree_idx = long int;
    using result_type   = mtcs_results;
    Game_Encode_Type encoded_game_state;
    flat_tree_idx    parent_idx;
    result_type      results;
};

template <class Game_Board>
class mtcs_search_engine
{
    // FIXME Remove public
public:
    using game_state_type                = Game_Board;
    using valid_moves_container          = typename Game_Board::valid_moves_container;
    using action_type                    = typename Game_Board::action_type;
    using encode_type                    = typename Game_Board::encode_type;
    using flat_tree_node                 = mtcs_flat_tree_node<encode_type>;
    using flat_tree_idx                  = typename flat_tree_node::flat_tree_idx;
    using points_type                    = typename flat_tree_node::result_type::points_type;
    using leaf_node_idx                  = size_t;
    using player_type                    = typename Game_Board::player_repr_type;
    using sampling_states_container_type = std::vector<flat_tree_node>;
    using leaf_nodes_container_type      = std::vector<flat_tree_idx>;


private:
    static constexpr s_Parent_tree_node = mtcs_flat_tree_node{ m_Initial_game_board.encode(), -1, {} };
    enum game_result_type
    {
        win,
        loss,
        tie
    };

public:
    mtcs_search_engine(const game_state_type& game_state) :
        m_Initial_game_board(game_state),
        m_Monte_carlo_sampling{ s_Parent_tree_node },
        m_Target_player{ m_Initial_game_board.current_player() }
    {
        assert(m_Initial_game_board.any_moves_left());
        std::cout << "Here!\n";
    }

    mtcs_search_engine() :
        m_Initial_game_board{ game_state_type{} },
        m_Monte_carlo_sampling{ s_Parent_tree_node }
    {
    }

public:
    /**
     * \brief Selects a the flat tree index of a selected leaf node
     * \return index of a leaf node (selected to be the next parent)
     */
    flat_tree_idx selection()
    {
        const leaf_node_idx leaf_idx           = random::randsize_t(0, m_Leaf_nodes_idx.size() - 1);
        const auto          selected_state_idx = m_Leaf_nodes_idx[leaf_idx];

        // remove element from leaf nodes
        std::swap(m_Leaf_nodes_idx[leaf_idx], m_Leaf_nodes_idx.back());
        m_Leaf_nodes_idx.pop_back();

        return selected_state_idx;
    }

    flat_tree_idx expansion(const flat_tree_idx next_parent_idx)
    {
        auto parent_game_state = game_state_type::decode(m_Monte_carlo_sampling[next_parent_idx].encoded_game_state);
        flat_tree_idx next_leaf_idx = m_Monte_carlo_sampling.size();
        const auto    valid_moves   = parent_game_state.get_valid_moves();

        const auto leaf_nodes_initial_size = m_Leaf_nodes_idx.size();
        size_t     added_leaves            = 0;

        for (auto move : valid_moves)
        {
            if (!move)
                continue;

            parent_game_state.make_move(move);
            m_Monte_carlo_sampling.push_back({ parent_game_state.encode(), next_parent_idx, {} });

            if (!parent_game_state.winning_move(move))
            {
                m_Leaf_nodes_idx.push_back(next_leaf_idx);
                ++added_leaves;
            }
            else
            {
                backpropagation(next_leaf_idx, parent_game_state.previous_player() == m_Target_player ? won() : lost());
            }
            ++next_leaf_idx;
            parent_game_state.undo_move(move);
        }

        if (added_leaves == 0)
            return m_Leaf_nodes_idx.back();

        // TODO Generalize rng type
        const auto offset = random::randsize_t(0, added_leaves - 1);
        return m_Leaf_nodes_idx[leaf_nodes_initial_size + offset];
    }

    game_result_type simulation(const flat_tree_idx game_state_idx)
    {
        auto& selected_node     = m_Monte_carlo_sampling[game_state_idx];
        auto  sample_game_state = game_state_type::decode(selected_node.encoded_game_state);

        while (sample_game_state.any_moves_left())
        {
            const auto selected_move = sample_game_state.select_random_move();
            sample_game_state.make_move(selected_move);
            if (sample_game_state.winning_move(selected_move))
                return m_Target_player == sample_game_state.previous_player() ? won() : lost();
        }
        return tied();
    }

    void backpropagation(flat_tree_idx game_state_idx, const game_result_type result)
    {
        const auto trial_score = points_increment(result);
        std::cout << game_state_idx;
        while (game_state_idx > -1)
        {
            m_Monte_carlo_sampling[game_state_idx].results.add_trial_score(trial_score);
            game_state_idx = m_Monte_carlo_sampling[game_state_idx].parent_idx;
            std::cout << "->" << game_state_idx;
        }
        std::cout << "\n";
    }

private:
    static constexpr game_result_type won()
    {
        return game_result_type::win;
    }
    static constexpr game_result_type tied()
    {
        return game_result_type::tie;
    }
    static constexpr game_result_type lost()
    {
        return game_result_type::loss;
    }
    static constexpr points_type points_increment(const game_result_type result)
    {
        switch (result)
        {
        case game_result_type::win:
            return 1;
        case game_result_type::tie:
            return 0.001f;
        case game_result_type::loss:
            return 0;
        default:
            // std::unreachable();
            throw std::runtime_error("Reached unreachable code");
        }
    }


private:
    // FIXME: Remove public
public:
    game_state_type                m_Initial_game_board;
    sampling_states_container_type m_Monte_carlo_sampling;
    leaf_nodes_container_type      m_Leaf_nodes_idx{ 0 };
    player_type                    m_Target_player{};
};
} // namespace mcts


#endif // ! MONTE_CARLO_TREE_SEARCH
