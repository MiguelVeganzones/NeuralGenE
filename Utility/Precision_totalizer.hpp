#ifndef PRECISION_TOTALIZER
#define PRECISION_TOTALIZER

#include <cmath>
#include <iostream>

class precision_totalizer
{
public:
    using signed_integral_type = int64_t;
    using floating_point_type  = long double;

private:
    floating_point_type  m_Fractional_accumulated_sum = 0;
    signed_integral_type m_Integral_accumulated_sum   = 0;

    floating_point_type m_Stddev  = 0;
    floating_point_type m_Average = 0;
    floating_point_type m_Min     = std::numeric_limits<floating_point_type>::max();
    floating_point_type m_Max     = std::numeric_limits<floating_point_type>::lowest();
    size_t              m_Samples = 0;

public:
    template <std::floating_point Floating_Point_Value>
    void add(const Floating_Point_Value d)
    {
        ++m_Samples;

        auto floating_point_value = static_cast<floating_point_type>(d);

        update_min(floating_point_value);
        update_max(floating_point_value);
        update_metrics(floating_point_value);

        floating_point_type d_i;
        floating_point_value = std::modf(floating_point_value, &d_i);

        m_Fractional_accumulated_sum += floating_point_value;
        m_Integral_accumulated_sum += static_cast<signed_integral_type>(d_i);

        reduce_fractional_reminder();
    }

    template <std::integral Integral_Type>
    void add(const Integral_Type n)
    {
        ++m_Samples;
        m_Integral_accumulated_sum += static_cast<Integral_Type>(n);
        const auto floating_point_value = static_cast<floating_point_type>(n);
        update_min(floating_point_value);
        update_max(floating_point_value);
        update_metrics(floating_point_value);
    }

    [[nodiscard]] auto samples() const
    {
        return m_Samples;
    }

    [[nodiscard]] auto get_floating_point_value() const
    {
        return m_Fractional_accumulated_sum;
    }

    [[nodiscard]] auto get_integral_value() const
    {
        return m_Integral_accumulated_sum;
    }

    [[nodiscard]] auto get_value() const
    {
        return m_Integral_accumulated_sum + m_Fractional_accumulated_sum;
    }

    [[nodiscard]] auto get_stddev() const
    {
        return m_Stddev;
    }

    [[nodiscard]] auto get_average() const
    {
        return m_Average;
    }

    [[nodiscard]] auto get_min() const
    {
        return m_Min;
    }

    [[nodiscard]] auto get_max() const
    {
        return m_Max;
    }

    [[nodiscard]] auto get_num_samples() const
    {
        return m_Samples;
    }

    void summary() const
    {
        std::cout << "\n_____Totalizer Summary_____";
        std::cout << "\nTotal samples:\t" << this->get_num_samples();
        std::cout << "\nTotal integral value:\t\t" << this->get_integral_value();
        std::cout << "\nRemainder fractional value:\t" << this->get_floating_point_value();
        std::cout << "\nMinimum value:\t\t" << this->get_min();
        std::cout << "\nMaximum value:\t\t" << this->get_max();
        std::cout << "\nAverage value:\t\t" << this->get_average();
        std::cout << "\nStandard deviation:\t" << this->get_stddev();
        std::cout << "\n___________________________\n\n";
    }

private:
    void update_metrics(const floating_point_type added_value)
    {
        const floating_point_type new_average = (m_Samples > 1) ? static_cast<floating_point_type>(m_Samples - 1) /
                    static_cast<floating_point_type>(m_Samples) * m_Average +
                added_value / static_cast<floating_point_type>(m_Samples)
                                                                : added_value;

        // https://math.stackexchange.com/questions/775391/can-i-calculate-the-new-standard-deviation-when-adding-a-value-without-knowing-t
        m_Stddev = std::pow(
            static_cast<floating_point_type>(m_Samples - 1) / static_cast<floating_point_type>(m_Samples) * m_Stddev *
                    m_Stddev +
                (added_value - new_average) * (added_value - m_Average) / static_cast<floating_point_type>(m_Samples),
            0.5
        );
        m_Average = new_average;
    }

    void update_min(const floating_point_type added_value)
    {
        if (added_value < m_Min)
            m_Min = added_value;
    }

    void update_max(const floating_point_type added_value)
    {
        if (added_value > m_Max)
            m_Max = added_value;
    }

    void reduce_fractional_reminder()
    {
        if (m_Fractional_accumulated_sum > 1.0)
        {
            --m_Fractional_accumulated_sum;
            ++m_Integral_accumulated_sum;
        }
        else if (m_Fractional_accumulated_sum < -1.0)
        {
            ++m_Fractional_accumulated_sum;
            --m_Integral_accumulated_sum;
        }
    }
};

#endif // PRECISION_TOTAIZER
