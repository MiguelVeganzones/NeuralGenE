#ifndef MINIMAX_TREE_SEARCH
#define MINIMAX_TREE_SEARCH

#include "Random.hpp"
#include <execution>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>

// #include "parallel_hashmap/phmap.h"

namespace minimax_tree_search
{
template <typename State_Type, typename Score_Type, typename Brain_Type>
    requires(requires {
        // Interface assumes that invalid actions can be represented by an
        // invalid state and thus checks for action validity when necessary.
        static_cast<bool>(std::declval<typename State_Type::action_type&>());
        // Hash required for the state encode type
        typename State_Type::encode_type_hasher;
        typename State_Type::encode_type_equal;
        // State type should define a way to be enocded
        std::declval<State_Type&>().encode();
    } && std::is_invocable_r_v<Score_Type, Brain_Type, State_Type>)
class minimax_search_engine
{
public:
    // TODO:
    // using container_type               = phmap::parallel_flat_hash_map;
    using state_type         = State_Type;
    using action_type        = typename State_Type::action_type;
    using score_type         = Score_Type;
    using encode_type        = typename State_Type::encode_type;
    using encode_type_hasher = typename State_Type::encode_type_hasher;
    using encode_type_equal  = typename State_Type::encode_type_equal;
    using search_state_container_type =
        std::unordered_map<encode_type, score_type, encode_type_hasher, encode_type_equal>;
    using brain_type              = Brain_Type;
    using valid_actions_container = typename State_Type::valid_actions_container;
    using player_type             = typename State_Type::player_repr_type;
    using scores_container_type   = std::array<score_type, State_Type::Size_x>;

    // TODO move to the end or move the rest to the beggining for consistency
private:
    // TODO: include an initial size as a ctor default parameter
    search_state_container_type m_Search_state{};
    state_type                  m_State;
    brain_type                  m_Brain;
    mutable std::size_t         count = 0;
    int                         m_Max_depth;
    player_type                 m_Target_player;

    // TODO Findout if initialisation is required
    mutable std::shared_mutex m_Search_state_mutex;

public:
    minimax_search_engine(const State_Type& state, const Brain_Type& brain, int max_depth) :
        m_State{ state },
        m_Brain{ brain },
        m_Max_depth{ max_depth },
        m_Target_player{ state.current_player() }
    {
    }

    // FIXME
    [[nodiscard]] auto minimax_search(State_Type state) -> std::optional<action_type>
    {
        if (!state.any_actions_left())
            return std::nullopt;
        auto valid_actions = state.get_valid_actions();
        auto action_scores = std::array<score_type, State_Type::Size_x>{};
        // TODO merge valid actions and their scores
        // for (const auto action : valid_actions)

        auto minimax_branch = [&](action_type action) -> void {
            action_scores[action.x] = static_cast<bool>(action)
                ? minimax_search_impl(state, action, 1, score_type::min(), score_type::max())
                : score_type::none();
        };

        std::for_each(std::execution::par, std::begin(valid_actions), std::end(valid_actions), minimax_branch);

        // for (std::size_t i = 0; i != valid_actions.size(); ++i)
        // {
        //     const auto action = valid_actions[i];
        //     std::cout << "Action: " << action << std::endl;
        //     if (!static_cast<bool>(action))
        //     {
        //         action_scores[i] = score_type::none();
        //         continue;
        //     }
        //     action_scores[i] = minimax_search_impl(state, action, 1, score_type::min(), score_type::max());
        //     action_scores[i] = non_abp_minimax_search_impl(state, action, 1);

        //     std::cout << minimax_search_impl(state, action, 1, score_type::min(), score_type::max()) << std::endl;
        //     std::cout << non_abp_minimax_search_impl(state, action, 1) << std::endl;
        // }

        print_search_state_info();
        for (unsigned i = 0; i != action_scores.size(); ++i)
        {
            std::cout << valid_actions[i] << ": " << action_scores[i] << '\n';
        }
        auto max = std::max_element(action_scores.begin(), action_scores.end());
        std::cout << "Max: " << *max << std::endl;
        return valid_actions[int(max - action_scores.begin())];
    }

    [[nodiscard]] auto minimax_search_impl(
        State_Type state, action_type action, int depth, score_type alpha, score_type beta
    ) -> score_type
    {
        count_one_up();
        assert(state.is_valid_action(action));
        state.perform_action(action);
        // std::cout << state.board_state() << std::endl;
        if (state.winning_action(action))
        {
            // std::cout << "here0\n";
            return game_finished_score(state.previous_player());
        }
        if (!state.any_actions_left())
        {
            // std::cout << "here1\n";
            return score_type::tied();
        }
        if (depth == m_Max_depth)
        {
            // std::cout << "here3\n";
            // Not storing this value reduces greately memory consumption and should increase speed.
            return m_Brain(state);
            // const auto score = m_Brain(state);
            // add_value(state, score);
            // return score;
        }
        if (const auto lookup_result = lookup_value(state); lookup_result.has_value())
        {
            // std::cout << "here2\n";
            // std::cout << state.board_state() << std::endl;
            return lookup_result.value();
        }

        const auto valid_actions = state.get_valid_actions();
        // TODO merge valid actions and their scores
        // for (const auto action : valid_actions)

        const bool max = state.current_player() == m_Target_player;
        max ? (alpha = score_type::min()) : (beta = score_type::max());

        for (const auto next_action : valid_actions)
        {
            // std::cout << "Action: " << next_action << std::endl;
            if (!static_cast<bool>(next_action))
            {
                continue;
            }
            const auto score = minimax_search_impl(state, next_action, depth + 1, alpha, beta);
            max ? (alpha = std::max(score, alpha)) : (beta = std::min(score, beta));

            if (beta <= alpha)
            {
                return max ? alpha : beta;
            }
        }

        const auto score = max ? alpha : beta;
        add_value(state, score);
        return score;
    }

    [[nodiscard]] auto non_abp_minimax_search_impl(State_Type state, action_type action, int depth) -> score_type
    {
        count_one_up();
        assert(state.is_valid_action(action));
        state.perform_action(action);
        // std::cout << state.board_state() << std::endl;
        if (state.winning_action(action))
        {
            // std::cout << "here0\n";
            return game_finished_score(state.previous_player());
        }
        if (!state.any_actions_left())
        {
            // std::cout << "here1\n";
            return score_type::tied();
        }
        if (auto lookup_result = lookup_value(state); lookup_result.has_value())
        {
            // std::cout << "here2\n";
            // std::cout << state.board_state() << std::endl;
            return lookup_result.value();
        }
        if (depth == m_Max_depth)
        {
            // std::cout << "here3\n";
            const auto score = m_Brain(state);
            add_value(state, score);
            return score;
        }

        auto valid_actions = state.get_valid_actions();
        // TODO merge valid actions and their scores
        // for (const auto action : valid_actions)
        scores_container_type scores{};
        int                   i = -1;
        for (auto next_action : valid_actions)
        {
            ++i;
            // std::cout << "Action: " << next_action << std::endl;
            if (!static_cast<bool>(next_action))
            {
                continue;
            }
            scores[i] = non_abp_minimax_search_impl(state, next_action, depth + 1);
        }

        auto score = best_score(scores, state.current_player());
        add_value(state, score);
        return score;
    }

private:
    auto lookup_value(const state_type& state) const -> std::optional<score_type>
    {
        std::shared_lock read_lock(m_Search_state_mutex);
        const auto       it = m_Search_state.find(state.encode());
        if (it != m_Search_state.end())
        {
            // std::cout << "dict: " << (int)it->second << std::endl;
            return it->second;
        }
        return std::nullopt;
    }

    auto add_value(const state_type& state, score_type score) -> void
    {
        std::unique_lock write_lock(m_Search_state_mutex);
        m_Search_state.try_emplace(state.encode(), score);
    }

    auto print_search_state_info() const -> void
        requires requires { typename search_state_container_type::hasher; }
    {
        std::unique_lock write_lock(m_Search_state_mutex);
        std::cout << "max_load factor: " << m_Search_state.max_load_factor()
                  << "\nBucket count: " << m_Search_state.bucket_count() << "\nItems in map: " << m_Search_state.size()
                  << "\nDictionary item size: " << sizeof(typename search_state_container_type::value_type) << '\n';

        // for (unsigned i = 0; i < m_Search_state.bucket_count(); ++i)
        // {
        //     std::cout << "bucket #" << i << " contains:\n";
        //     for (auto it = m_Search_state.cbegin(i); it != m_Search_state.cend(i); ++it)
        //     {
        //         std::cout << '[';
        //         for (auto v : it->first)
        //             std::cout << (int)v << ", ";
        //         std::cout << "]: " << it->second << '\n';
        //     }
        //     std::cout << '\n';
        // }
        std::cout << count << std::endl;
    }

    auto print_search_state_info() const -> void
    {
        std::unique_lock write_lock(m_Search_state);
        std::cout << "\nItems in map: " << m_Search_state.size() << '\n';

        for (const auto& [key, value] : m_Search_state)
        {
            std::cout << '[';
            for (auto v : key)
                std::cout << v << ", ";
            std::cout << "]: " << value << '\n';
        }
        std::cout << '\n';
        std::cout << count << std::endl;
    }

    auto count_one_up() const -> void
    {
        ++count;
    }

    [[nodiscard]] auto best_score(const scores_container_type& scores, player_type player) -> score_type
    {
        return (
            player == m_Target_player ? *std::max_element(scores.begin(), scores.end())
                                      : *std::min_element(scores.begin(), scores.end())
        );
    }

    [[nodiscard]] auto game_finished_score(player_type player) -> score_type
    {
        return score_type{ player == m_Target_player ? score_type::won() : score_type::lost() };
    }

}; // namespace minimax_tree_search
} // namespace minimax_tree_search


#endif // MINIMAX_TREE_SEARCH
