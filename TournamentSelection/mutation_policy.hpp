#ifndef MUTATION_POLICY_GENERATOR
#define MUTATION_POLICY_GENERATOR

#include "tournament_selection.hpp"

// FIXME gene replacement plocy?
namespace mutation_policy
{

template <typename T>
concept mutation_policy_generator_concept = requires(T t) {
    typename T::value_type;
    typename T::parameters_type;
    typename T::mutation_policy_type;

    {
        T::default_parameters()
    } -> std::same_as<typename T::parameters_type>;
    {
        T::generate(std::declval<typename T::parameters_type&>())
    } -> std::same_as<typename T::mutation_policy_type>;
    {
        std::declval<typename T::mutation_policy_type&>().get_parameters()
    } -> std::same_as<typename T::parameters_type>;
};

template <std::floating_point Value_Type>
class mutation_policy_generator
{
public:
    inline static constexpr std::size_t N = 6;

    using value_type             = Value_Type;
    using probability_type       = tournament_selection::floating_point_normalized_type<float>;
    using parameters_type        = std::array<probability_type, N>;
    using mutation_function_type = value_type (*)(value_type);

    class mutation_policy_type
    {
    public:
        using value_type      = mutation_policy_generator::value_type;
        using parameters_type = mutation_policy_generator::parameters_type;

    private:
        parameters_type m_Params;
        parameters_type m_Cummulative_params;

    public:
        mutation_policy_type(parameters_type const& params) noexcept :
            m_Params(params)
        {
            for (int i = 1; i != N; ++i)
            {
                m_Cummulative_params[i] = m_Params[i] + m_Cummulative_params[i - 1];
            }
        }

        mutation_policy_type() noexcept                                       = delete;
        mutation_policy_type(mutation_policy_type const&) noexcept            = default;
        mutation_policy_type(mutation_policy_type&&) noexcept                 = default;
        mutation_policy_type& operator=(mutation_policy_type const&) noexcept = default;
        mutation_policy_type& operator=(mutation_policy_type&&) noexcept      = default;
        ~mutation_policy_type() noexcept                                      = default;

        [[nodiscard]]
        auto get_parameters() const noexcept -> parameters_type
        {
            return m_Params;
        }

        [[nodiscard]]
        auto
        operator()(value_type f) const noexcept -> value_type
        {
            const auto r = random::randfloat();
            if (r >= m_Cummulative_params[5])
            {
                return f;
            }
            if (r < m_Cummulative_params[0])
            {
                return random::randfloat();
            }
            if (r < m_Cummulative_params[1])
            {
                return f * random::randnormal();
            }
            if (r < m_Cummulative_params[2])
            {
                return value_type{};
            }
            if (r < m_Cummulative_params[3])
            {
                return f += random::randnormal(random::randnormal(), random::randfloat());
            }
            if (r < m_Cummulative_params[4])
            {
                return random::randnormal(f, value_type(0.01));
            }
            if (r < m_Cummulative_params[5])
            {
                return f * value_type(0.9999);
            }
            assert_unreachable();
        }
    };

    [[nodiscard]]
    static auto default_parameters() noexcept -> parameters_type
    {
        return s_Default_params;
    }

    inline static constexpr probability_type p0               = probability_type(0.0005f);
    inline static constexpr parameters_type  s_Default_params = { p0, p0, p0, p0, p0, p0 };

public:
    [[nodiscard]]
    static auto generate(parameters_type const& params) noexcept -> mutation_policy_type
    {
        // TODO mutate params
        return mutation_policy_type(params);
    }
};
} // namespace mutation_policy
#endif // MUTATION_POLICY_GENERATOR