#ifndef REPRODUCTION_POLICY
#define REPRODUCTION_POLICY

#include "Random.hpp"
#include "error_handling.hpp"
#include "static_matrix.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <type_traits>

namespace reproduction_policy
{

struct parent
{
    std::uint8_t index;
};

struct asexual_reproduction_parent
{
    parent p;
};

struct sexual_reproduction_parents
{
    parent a;
    parent b;
}

template <std::uint16_t Population_Size>
class mixed_reproduction_policy

{
public:
    inline static constexpr auto s_Populaiton_Size = Population_Size;

    using parent_type                   = parent; // Maybe make parent a template?
    using chosen_parent_type            = std::variant<parent, std::pair<parent, parent>>;
    using chosen_parents_container_type = std::array<chosen_parent_type, s_Population_Size>;

    inline [[nodiscard]]
    auto select_parents(
        ga_sm::static_matrix<float, s_Populaiton_Size, s_Populaiton_Size> const&,
        std::array<float, s_Populaiton_Size> const& fitness_scores
    ) -> std::array<parent_selection_result>
    {
        // TODO
    }
};

// TODO : Require N % 2 == 0 in only sexuaql reproduction policies

} // namespace reproduction_policy

#endif // REPRODUCTION_POLICY
