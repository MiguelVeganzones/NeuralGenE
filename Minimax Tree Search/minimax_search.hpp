#ifndef MINIMAX_TREE_SEARCH
#define MINIMAX_TREE_SEARCH

#include "Random.h"
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <shared_mutex>

// #include "parallel_hashmap/phmap.h"

namespace minimax_tree_search
{
template <typename State_Type, typename Score_Type, typename Brain>
    requires(requires {
                 // Interface assumes that invalid actions can be represented by an
                 // invalid state and thus checks for action validity when necessary.
                 static_cast<bool>(std::declval<typename State_Type::action_type>());
                 // Hash required for the state encode type
                 typename State_Type::hash_function;
             } && std::is_invocable_r_v<Score_Type, Brain, State_Type>)
class minimax_search_engine
{
public:
    // TODO:
    // using container_type               = phmap::parallel_flat_hash_map;
    using state_type                  = State_Type;
    using action_type                 = typename State_Type::action_type;
    using score_type                  = Score_Type;
    using encode_type                 = typename State_Type::encode_type;
    using hash                        = typename State_Type::hash_function;
    using search_state_container_type = std::unordered_map<encode_type, score_type, hash>;
    using brain_type                  = Brain;

    // TODO move to the end or move the rest to the beggining
private:
    // TODO: include an initial size as a ctor default parameter
    search_state_container_type m_Search_state{};
    state_type                  m_State;
    brain_type                  m_Brain;

    // TODO Findout if initialisation is required
    std::shared_mutex m_Search_state_mutex;

public:
    minimax_search_engine(const State_Type& state, const Brain& brain) :
        m_State{ state },
        m_Brain{ brain }
    {
    }

    [[nodiscard]] auto minimax_search(const State_Type& state) -> std::optional<action_type>
    {
        if (!state.any_actions_left())
            return std::nullopt;
        auto valid_actions = state.get_valid_actions();
        for (const auto action : valid_actions)
        {
            if (!static_cast<bool>(action))
                continue;
            auto current_state = state.perform_action(action);
            if (auto lookup_result = lookup_value(current_state); lookup_result.has_value)
            {
                return lookup_result.value;
            }
        }
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
}; // namespace minimax_tree_search
} // namespace minimax_tree_search


#endif // MINIMAX_TREE_SEARCH
