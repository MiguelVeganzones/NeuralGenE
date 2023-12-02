#include "c4_board.hpp"
#include "mcts.hpp"
#include <iostream>

int main()
{
    random::init();

    constexpr size_t M = 4;
    constexpr size_t N = 4;
    constexpr size_t K = 2;

    std::cout << "MCTS main\n\n";

    using board_type = c4_board::board<M, N, 3, K>;
    board_type b{};

    // while (b.any_moves_left() && random::randfloat() > 0.05f)
    //{
    //     b.make_move(b.select_random_move());
    //     std::cout << b.board_state() << std::endl;
    // }

    // std::cout << b.board_state() << std::endl;

    // if (!b.any_moves_left())
    //     return 0;

    mcts::mcts_search_engine initial_state(b);

    for (int i = 0; i < 100; ++i)
    {
        auto selected = initial_state.selection();
        if ((initial_state.m_Leaf_nodes_idx.size() == 0) && i != 0)
        {
            break;
        }
        auto       playout_state = initial_state.expansion(selected);
        const auto result        = initial_state.simulation(playout_state);
        initial_state.backpropagation(playout_state, result);
        std::cout << initial_state.m_Monte_carlo_sampling[0].results.samples << std::endl;
    }

    for (int i = 0; i < 10; ++i)
    {
        auto e2 = initial_state.m_Monte_carlo_sampling[i];

        std::cout << "Board: " << i << std::endl;

        std::cout << decltype(initial_state)::game_state_type::decode(e2.encoded_game_state).board_state();

        std::cout << e2.results.points << '/' << e2.results.samples << "\n";
    }
    std::cout << std::endl;


    return EXIT_SUCCESS;
}