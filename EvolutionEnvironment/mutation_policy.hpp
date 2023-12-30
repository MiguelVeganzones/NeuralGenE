#ifndef MUTATION_POLICY_GENERATOR
#define MUTATION_POLICY_GENERATOR

#include "Random.hpp"
#include "error_handling.hpp"
#include <array>
#include <concepts>
#include <type_traits>

// FIXME gene replacement plocy?
namespace mutation_policy_
{

template <std::floating_point Value_Type, std::size_t N>
class mutation_policy
{
public:
    using value_type      = Value_Type;
    using parameters_type = std::array<value_type, N>;

public:
    mutation_policy(parameters_type const& params) noexcept :
        m_Params(params)
    {
        update_cummulative_params();
    }

    mutation_policy(value_type p) noexcept
    {
        m_Params.fill(p);
        update_cummulative_params();
    }

    mutation_policy() noexcept                                  = delete;
    mutation_policy(mutation_policy const&) noexcept            = default;
    mutation_policy(mutation_policy&&) noexcept                 = default;
    mutation_policy& operator=(mutation_policy const&) noexcept = default;
    mutation_policy& operator=(mutation_policy&&) noexcept      = default;
    ~mutation_policy() noexcept                                 = default;

    [[nodiscard]]
    auto get_parameters() const noexcept -> parameters_type
    {
        return m_Params;
    }

    auto set_parameters(parameters_type const& params) -> void
    {
        m_Params = params;
        update_cummulative_params();
    }

    auto set_base_probability(value_type p) -> void
    {
        m_Params = [](value_type p_) -> parameters_type {
            parameters_type params;
            params.fill(p_);
            return params;
        }(p);
        update_cummulative_params();
    }

    [[nodiscard]]
    auto
    operator()(value_type f) const noexcept -> value_type
    {
        const auto r = m_Random_engine.randfloat();
        if (r < m_Cummulative_params[0])
        {
            f *= value_type(1.05);
        }
        else if (r < m_Cummulative_params[1])
        {
            f += m_Random_engine.randnormal(0.f, 0.15f);
        }
        else if (r < m_Cummulative_params[2])
        {
            f += m_Random_engine.randnormal(0.f, 0.3f);
        }
        else if (r < m_Cummulative_params[3])
        {
            f = m_Random_engine.randnormal(0.f, 0.2f);
        }
        else if (r < m_Cummulative_params[4])
        {
            f = m_Random_engine.randnormal(0.f, 0.4f);
        }
        else
        {
            return f * value_type(0.9999f);
        }
        return std::clamp(f, value_type(-1), value_type(1));
    }

private:
    auto update_cummulative_params() -> void
    {
        std::partial_sum(
            std::begin(m_Params),
            std::end(m_Params),
            std::begin(m_Cummulative_params)
        );
    }

private:
    mutable random_::random m_Random_engine{};
    parameters_type         m_Params;
    parameters_type         m_Cummulative_params;
};

} // namespace mutation_policy_
#endif // MUTATION_POLICY_GENERATOR