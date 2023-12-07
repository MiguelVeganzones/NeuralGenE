#ifndef FITNESS_CALCULATOR
#define FITNESS_CALCULATOR

#include "tournament_selection.hpp"

namespace fitness_calculator
{

// TODO Restrict template
// template <
//     typename generation_container_type,
//     typename scores_container_type,
//     typename reproduction_probabilities_container_type>
template <typename Agent>
struct fitness_calculator
{
    using agent_type = Agent; // std::remove_cvref_t<typename
                              // generation_container_type::value_type>;
    using probability_type =
        tournament_selection::floating_point_normalized_type<float>;

    inline static constexpr float distance_threshold = 0.1f; // TODO

    // auto operator()(generation_container_type const&, scores_container_type
    // const&)
    // {
    //     reproduction_probabilities_container_type();
    // }

    template <std::floating_point R, std::size_t N>
        requires(N > 1)
    [[nodiscard]]
    static auto generation_variability(std::array<agent_type, N> const& agents)
        -> ga_sm::static_matrix<R, N, N>
    {
        ga_sm::static_matrix<R, N, N> distance_matrix{};
        for (size_t j = 0; j != N - 1; ++j)
        {
            for (size_t i = j + 1; i != N; ++i)
            {
                const auto distance = agent_distance<R>(
                    agents[j],
                    agents[i],
                    [](float a, float b) { return std::pow(a - b, 2); }
                );
                distance_matrix[j, i] = distance;
                distance_matrix[i, j] = distance;
            }
        }
        return distance_matrix;
    }

    template <std::size_t N>
        requires(N > 1)
    //[[nodiscard]]
    static auto generation_fitness(
        std::array<agent_type, N> const& agents,
        std::array<float, N>             generation_loss
    ) -> std::array<probability_type, N>
    {
        const auto distance_matrix = generation_variability(agents);

        std::array<std::size_t, N> indexer{};
        std::iota(indexer.begin(), indexer.end(), 0);

        std::sort(
            indexer.begin(),
            indexer.end(),
            [&generation_loss](const std::size_t idx_1, const std::size_t idx_2)
                -> bool {
                return generation_loss[idx_1] < generation_loss[idx_2];
            }
        );

        std::array<probability_type, N> scores{};
        std::array<bool, N>             done{ false };
        const auto max_loss = std::ranges::max(generation_loss);

        for (auto j : indexer)
        {
            if (done[j])
                continue;
            scores[j] = 1 - (generation_loss[j] / max_loss);
            for (std::size_t i = 0; i != N; ++i)
            {
                if (i == j || done[i])
                {
                    continue;
                }
                if (distance_matrix[j, i] < distance_threshold)
                {
                    done[i] = true;
                    continue;
                }
            }
        }
        for (auto e : scores)
            std::cout << e << ' ';
        std::cout << std::endl;
        return scores;
    }
};

} // namespace fitness_calculator

#endif // FITNESS_CALCULATOR