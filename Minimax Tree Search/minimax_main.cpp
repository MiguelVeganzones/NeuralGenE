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

    constexpr auto fn = score_functions::score_functions<double, std::uint16_t>::
        choose_function<score_functions::Identifiers::Weighted_normalized_score, 3, 1, 0>();

    constexpr auto AF_relu = matrix_activation_functions::Identifiers::GELU;
    constexpr auto AF_tanh = matrix_activation_functions::Identifiers::Sigmoid;
    auto           SM =
        score_function_objects::score_function_object<decltype(fn), std::uint16_t, std::uint16_t, std::uint16_t>(fn);


    using SM_t = decltype(SM);

    [[maybe_unused]] constexpr Layer_Signature a1{ 1, AF_relu };
    [[maybe_unused]] constexpr Layer_Signature a1_tanh{ 1, AF_tanh };
    [[maybe_unused]] constexpr Layer_Signature a9{ 9, AF_relu };

    using NET        = static_neural_net<float, 1, a1, a9, a9, a1_tanh>;
    using brain_type = ga_neural_model::brain<NET, SM_t>;
    auto                                                                                       brain = brain_type{};
    [[maybe_unused]] minimax_tree_search::minimax_search_engine<board_type, float, brain_type> mmse(b, brain);
    std::cout << "Minimax search\n";


    return EXIT_SUCCESS;
}