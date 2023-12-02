#ifndef SCORE_FUNCTIONS
#define SCORE_FUNCTIONS

#include <concepts>
#include <type_traits>

namespace score_function_objects
{

template <typename Fn_Obj>
concept Score_Function = requires(Fn_Obj fn) {
    typename Fn_Obj::return_type;
    typename Fn_Obj::input_type;
    {
        std::invoke(fn, std::declval<typename Fn_Obj::input_type&>())
    } -> std::convertible_to<typename Fn_Obj::output_type>;
};

template <typename R, typename Fn, typename Arg>
    requires std::is_invocable_r_v<R, Fn, Arg>
class score_function
{
public:
    using function_type = Fn;
    using input_type    = Arg;
    using output_type   = R;

    Fn m_Fn;

    score_function(Fn&& fn) noexcept :
        m_Fn(std::forward<Fn>(fn))
    {
    }

    score_function() noexcept                                 = delete;
    score_function(const score_function&) noexcept            = default;
    score_function(score_function&&) noexcept                 = default;
    score_function& operator=(const score_function&) noexcept = default;
    score_function& operator=(score_function&&) noexcept      = default;
    ~score_function() noexcept                                = default;

    template <typename Input_Type>
        requires std::same_as<std::remove_cvref_t<input_type>, std::remove_cvref_t<Input_Type>> ||
        std::is_same<std::decay_t<input_type>, std::decay_t<Input_Type>>
    [[nodiscard]] auto operator()(Input_Type&& input) noexcept -> output_type
    {
        return std::invoke(m_Fn, std::forward<decltype(input)>(input));
    }
};


} // namespace score_function_objects

#endif // !SCORE_FUNCTIONS
