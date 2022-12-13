#ifndef SCORE_FUNCTIONS
#define SCORE_FUNCTIONS

#include <stdexcept>

namespace score_functions
{

enum Identifiers
{
    Weighted_normalized_score
};

template <typename Return_Type, typename Input_Type>
class score_functions
{
public:
    using f_ptr = Return_Type (*)(Input_Type, Input_Type, Input_Type);

private:
    template <int Wins_Weight = 3, int Ties_Weight = 1, int Loses_Weight = 0>
    inline static constexpr Return_Type weighted_normalized_score(Input_Type wins, Input_Type ties, Input_Type loses)
    {
        const auto count = wins + ties + loses;

        if (count == 0)
            return 0;

        const auto d_count = static_cast<double>(count);

        return (wins / d_count) * Wins_Weight + (ties / d_count) * Ties_Weight + (loses / d_count) * Loses_Weight;
    }

public:
    template <Identifiers Identifier, int... Args>
        requires(sizeof...(Args) == 3)
    inline static constexpr f_ptr choose_function()
    {
        if constexpr (Identifier == Weighted_normalized_score)
            return weighted_normalized_score<Args...>;
        throw std::invalid_argument("Unexpected activation function identifier");
    }
};
} // namespace score_functions

#endif // !SCORE_FUNCTIONS
