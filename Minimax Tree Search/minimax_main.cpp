#include "Stopwatch.h"
#include "minimax_search.hpp"
#include <c4_board.hpp>
#include <iostream>
#include <limits>
#include <neural_model.hpp>

struct score_type
{
    using value_type = float;
    // inline static constexpr auto Min = std::numeric_limits<value_type>::lowest();
    // inline static constexpr auto Max = std::numeric_limits<value_type>::max();
    // inline static constexpr auto Avg = static_cast<value_type>((Max - Min) / 2);
    inline static constexpr auto Min = 0.f;
    inline static constexpr auto Max = 1.f;
    inline static constexpr auto Avg = 0.5f;
    inline static constexpr auto NaN = std::numeric_limits<value_type>::quiet_NaN();

    inline static constexpr auto won() -> score_type
    {
        return score_type{ Max };
    }

    inline static constexpr auto tied() -> score_type
    {
        return score_type{ Avg };
    }

    inline static constexpr auto lost() -> score_type
    {
        return score_type{ Min };
    }

    inline static constexpr auto none() -> score_type
    {
        return score_type{ NaN };
    }

    inline static constexpr auto min() -> decltype(lost())
    {
        return lost();
    }

    inline static constexpr auto max() -> decltype(won())
    {
        return won();
    }

    value_type value;

    constexpr score_type(value_type v = 0.f) :
        value{ cx_helper_func::cx_max(cx_helper_func::cx_min(v, Max), Min) }
    {
    }

    operator value_type() const
    {
        return value;
    }

    auto operator<<(std::ostream& os) const -> std::ostream&
    {
        os << value;
        return os;
    }

    auto operator>(score_type other) const -> bool
    {
        return value > other.value;
    }
};

int main()
{
    random::init();

    constexpr size_t M = 6;
    constexpr size_t N = 7;
    constexpr size_t K = 2;

    using board_t = c4_board::board<M, N, 4, K>;

    board_t b{};

    using namespace ga_snn;
    using namespace ga_sm;

    [[maybe_unused]] constexpr auto AF_relu = matrix_activation_functions::Identifiers::GELU;
    [[maybe_unused]] constexpr auto AF_tanh = matrix_activation_functions::Identifiers::Sigmoid;

    [[maybe_unused]] constexpr Layer_Signature a1{ 1, AF_relu };
    [[maybe_unused]] constexpr Layer_Signature a1_tanh{ 1, AF_tanh };
    [[maybe_unused]] constexpr Layer_Signature a9{ 9, AF_relu };
    [[maybe_unused]] constexpr Layer_Signature a25{ 25, AF_relu };
    [[maybe_unused]] constexpr Layer_Signature a42{ M * N, AF_relu };
    [[maybe_unused]] constexpr Layer_Signature a16{ 16, AF_relu };

    // using NET = static_neural_net<float, 1, a42, a9, a1_tanh>;

    // using preprocessor  = data_processor::c4_data_preprocessor<encoders::tristate_encoder>;
    // using postprocessor = data_processor::uniform_normalized_random;

    // using brain_t = ga_neural_model::brain<NET, preprocessor, postprocessor, score_type>;
    // brain_t brain(random::randnormal, 0, 1);
    auto brain = [](board_t const& board) -> float {
        using hasher = typename board_t::encode_type_hasher;
        hasher            hash{};
        auto              value = hash(board.encode());
        const std::size_t max   = 10000; // std::numeric_limits<std::size_t>::max();
        value                   = value * (value + 1) * (value + 2) % max;
        return static_cast<float>(static_cast<double>(value) / static_cast<double>(max));
    };
    using brain_t = decltype(brain);

    using minimax_search_t = minimax_tree_search::minimax_search_engine<board_t, score_type, brain_t>;
    std::cout << "Minimax search\n";

    stopwatch global{};
    for (int i = 1; i != 13; ++i)
    {
        stopwatch                         s{};
        [[maybe_unused]] minimax_search_t mmse(b, brain, i);
        [[maybe_unused]] auto             move = mmse.minimax_search(b);
        std::cout << move.value() << std::endl;
    }


    return EXIT_SUCCESS;
}