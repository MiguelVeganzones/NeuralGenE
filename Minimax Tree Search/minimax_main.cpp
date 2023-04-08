#include "minimax_search.hpp"
#include <c4_board.hpp>
#include <iostream>
#include <neural_model.hpp>

int main()
{
    // random::init();

    constexpr size_t M = 6;
    constexpr size_t N = 7;
    constexpr size_t K = 2;

    using board_type = c4_board::board<M, N, 3, K>;

    board_type b{};

    using namespace ga_snn;
    using namespace ga_sm;

    constexpr auto AF_relu = matrix_activation_functions::Identifiers::GELU;
    constexpr auto AF_tanh = matrix_activation_functions::Identifiers::Sigmoid;

    [[maybe_unused]] constexpr Layer_Signature a1{ 1, AF_relu };
    [[maybe_unused]] constexpr Layer_Signature a1_tanh{ 1, AF_tanh };
    [[maybe_unused]] constexpr Layer_Signature a9{ 9, AF_relu };
    [[maybe_unused]] constexpr Layer_Signature a25{ 25, AF_relu };
    [[maybe_unused]] constexpr Layer_Signature a42{ 42, AF_relu };
    [[maybe_unused]] constexpr Layer_Signature a16{ 16, AF_relu };

    using NET = static_neural_net<float, 1, a1, a9, a1_tanh>;

    using preprocessor  = data_processor::scalar_converter;
    using postprocessor = data_processor::scalar_converter;

    using brain_t = ga_neural_model::brain<NET, preprocessor, postprocessor, NET::value_type>;

    brain_t brain(random::randnormal, 0, 1);

    [[maybe_unused]] minimax_tree_search::minimax_search_engine<board_type, NET::value_type, brain_t> mmse(b, brain);
    std::cout << "Minimax search\n";


    return EXIT_SUCCESS;
}