#include <iostream>
#include "mcts.hpp"
#include "../Connect Four/c4_board.hpp"
//#include "c4_board.hpp"


int main()
{
    random::init();

    constexpr size_t M = 4;
    constexpr size_t N = 4;
    constexpr size_t K = 2;

    std::cout << "MCTS main\n\n";

    using board_type = c4_board::board<M, N, 3, K>;
    board_type b{};

    //while (b.any_moves_left() && random::randfloat() > 0.05f)
    //{
    //    b.make_move(b.select_random_move());
    //    std::cout << b.board_state() << std::endl;
    //}

    //std::cout << b.board_state() << std::endl;

    //if (!b.any_moves_left())
    //    return 0;

    mcts::initial_state initial_state(b);

    for (int i = 0; i < 100000000; ++i)
    {
        auto selected = initial_state.selection();
        if ((initial_state.m_Leaf_nodes_idx.size() != 0) || i == 0)
        {
            auto playout_state = initial_state.expansion(selected);
            auto result        = initial_state.simulation(playout_state);
            initial_state.backpropagation(playout_state, result);
        }
        else
        {
            break;
        }
    }

    for (size_t i = 0; i < 40; ++i)
    {
        auto e2 = initial_state.m_Monte_carlo_sampling[i];

        std::cout << decltype(initial_state)::game_state::decode(e2.encoded_game_state).board_state();

        std::cout << e2.results.points << '/' << e2.results.samples << "\n";
        
    }
    std::cout << std::endl;


    return EXIT_SUCCESS;
}