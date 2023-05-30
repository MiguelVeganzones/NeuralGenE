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
                                 std::invoke(fn, std::declval<typename Fn_Obj::input_type>())
                                 } -> std::convertible_to<typename Fn_Obj::output_type>;
                         };

template <typename R, typename Fn, typename Arg>
    requires std::is_invocable_r_v<R, Fn, Arg>
class score_function
{
    using function_type = Fn;
    using input_type    = Arg;
    using output_type   = R;

    auto operator()(const input_type& input) noexcept -> output_type
    {
        // return std::invoke(m_Fn, input);
        return 0;
    }
};


} // namespace score_function_objects

#endif // !SCORE_FUNCTIONS
