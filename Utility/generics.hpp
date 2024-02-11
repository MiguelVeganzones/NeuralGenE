#ifndef GENERICS_UTILITY
#define GENERICS_UTILITY

#include <cassert>
#include <concepts>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <vector>

namespace generics
{
namespace containers
{

template <typename T>
class restricted
{
public:
    restricted(T min, T max, T value) noexcept :
        min_{ min },
        max_{ max },
        value_{ value }
    {
    }

    restricted()                             = delete;
    restricted(restricted const&)            = default;
    restricted(restricted&&)                 = default;
    restricted& operator=(restricted const&) = default;
    restricted& operator=(restricted&&)      = default;

    [[nodiscard]]
    auto get_value() const noexcept -> T
    {
        return value_;
    }

    [[nodiscard]]
    auto min() const noexcept -> T
    {
        return min_;
    }

    [[nodiscard]]
    auto max() const noexcept -> T
    {
        return max_;
    }

    auto set_value(T value) noexcept -> void
    {
        value_ = clip(value);
    }

    auto increment(T delta = T(1)) -> void
    {
        set_value(get_value() + delta);
    }

    auto decrement(T delta = T(1)) -> void
    {
        set_value(get_value() - delta);
    }

private:
    T clip(T value)
    {
        return value > max_ ? max_ : (value < min_ ? min_ : value);
    }

private:
    T min_;
    T max_;
    T value_;
};
} // namespace containers

namespace algorithms
{
template <std::ranges::input_range Input_Range, typename Comp>
    requires std::is_invocable_r_v<
        bool,
        Comp,
        typename Input_Range::value_type,
        typename Input_Range::value_type>
[[nodiscard]]
auto argmax(Input_Range const& v, Comp&& comp)
{
    return std::distance(
        std::cbegin(v), std::ranges::max_element(v, std ::forward<Comp>(comp))
    );
}

template <std::ranges::input_range Input_Range, typename Comp>
    requires std::is_invocable_r_v<
        bool,
        Comp,
        typename Input_Range::value_type,
        typename Input_Range::value_type>
[[nodiscard]]
auto argmin(Input_Range const& v, Comp&& comp)
{
    return std::distance(
        std::cbegin(v), std::ranges::min_element(v, std ::forward<Comp>(comp))
    );
}

[[nodiscard]]
auto top_n_indeces(std::ranges::input_range auto const& v, unsigned int n)
    -> std::vector<int>
{
    assert(v.size() >= n && n > 0);
    std::vector<int> ret(n);
    // initialize top_n to first n elements
    std::iota(ret.begin(), ret.end(), 0);
    auto sort = [&v](auto&& vec) {
        std::ranges::sort(vec, [&v](auto&& a, auto&& b) {
            return v[a] < v[b];
        });
    };
    sort(ret);

    for (auto i = n; i != v.size(); ++i)
    {
        // TODO template operation
        if (v[ret[0]] < v[i])
        {
            ret[0] = i;
            sort(ret);
        }
    }
    return ret;
}

inline static auto L2_norm = []<std::floating_point R>(R a, R b) -> R {
    const auto d = (a - b);
    return static_cast<R>(d * d);
};

inline static auto L1_norm = []<std::floating_point R>(R a, R b) -> R {
    return static_cast<R>(std::abs(a - b));
};

} // namespace algorithms

namespace traits
{
template <auto>
struct require_constexpr;

template <typename R>
constexpr auto is_constexpr_size(R&& r) -> bool
{
    return requires {
        typename require_constexpr<std::ranges::size(std::forward<R>(r))>;
    };
}
} // namespace traits

} // namespace generics

#endif // GENERICS_UTILITY