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

template <
    std::uint16_t         Population_Size,
    std::floating_point_t Score_Type,
    std::floatng_point_t  Distance_Type>
class mixed_reproduction_policy

{
public:
    inline static constexpr auto s_Populaiton_Size = Population_Size;

    using score_type    = Score_Type;
    using distance_type = Distance_Type;
    using chosen_parent_type =
        std::variant<asexual_reproduction_parent, sexual_reproduction_parents>;
    using chosen_parents_container_type =
        std::array<chosen_parent_type, s_Population_Size>;

    [[nodiscard]]
    inline auto select_parents(
        ga_sm::
            static_matrix<float, s_Populaiton_Size, s_Populaiton_Size> const&,
        std::array<float, s_Populaiton_Size> const& fitness_scores
    ) -> std::array<parent_selection_result>
    {
        // TODO
    }

private:
    score_type    best_socre_;
    distance_type mean_distance_;
    float         fitness_score_importance_;
    float         distance_score_importance_;
};

// TODO : Require N % 2 == 0 in only sexual reproduction policies

} // namespace reproduction_policy

#endif // REPRODUCTION_POLICY
