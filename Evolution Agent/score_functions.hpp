#ifndef SCORE_FUNCTIONS
#define SCORE_FUNCTIONS

#include <stdexcept>

namespace score_function_objects
{

template <typename Fn, typename... Args>
    requires std::is_invocable_v<Fn, Args...>
class score_function_object
{
public:
    constexpr explicit score_function_object(Fn fn, Args... args) : m_Fn{ fn }, m_Internal_state{ args... }
    {
    }

    auto operator()() const
    {
        return std::apply(m_Fn, m_Internal_state);
    }

private:
    Fn                  m_Fn;
    std::tuple<Args...> m_Internal_state;
};

template <typename Fn, typename T>
    requires std::is_invocable_v<Fn, T, T, T>
class score_function_object<Fn, T, T, T>
{
public:
    constexpr explicit score_function_object(Fn fn) :
        m_Fn{ fn }, m_Wins{ 0 }, m_Ties{ 0 }, m_Loses{ 0 }
    {
    }

    auto operator()() const
    {
        return std::invoke(m_Fn, m_Wins, m_Ties, m_Loses);
    }

    void won()
    {
        ++m_Wins;
    }
    void tied()
    {
        ++m_Ties;
    }
    void lost()
    {
        ++m_Loses;
    }

    [[nodiscard]] auto wins() const
    {
        return m_Wins;
    }
    [[nodiscard]] auto ties() const
    {
        return m_Ties;
    }
    [[nodiscard]] auto losses() const
    {
        return m_Loses;
    }

private:
    Fn m_Fn;
    T  m_Wins;
    T  m_Ties;
    T  m_Loses;
};

/* -------------------- Concept -------------------- */

template <typename Fn, typename... Args>
void score_function_object_dummy(score_function_object<Fn, Args...>)
{
}

template <typename T>
concept score_function_object_type = requires { score_function_object_dummy(std::declval<T>()); };

} // namespace score_function_objects

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
    inline static constexpr Return_Type weighted_normalized_score(Input_Type wins,
                                                                  Input_Type ties,
                                                                  Input_Type loses) noexcept
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
