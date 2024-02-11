#ifndef PROGRESSBAR
#define PROGRESSBAR

#include <cassert>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>

namespace progressbar_
{

class progressbar
{
public:
    inline static constexpr auto total_progress = 100;
    inline static constexpr auto size_x         = 25;
    inline static constexpr auto progress_per_tick =
        static_cast<float>(total_progress / size_x);
    inline static constexpr auto open_char  = '[';
    inline static constexpr auto close_char = ']';
    inline static constexpr auto fill_char  = '#';
    inline static constexpr auto empty_char = ' ';

    template <typename... To_Print>
    auto print(
        int              progress,
        std::string_view leader = "\0",
        To_Print... printables
    ) noexcept -> void
    {
        m_Progress = progress;
        const auto progress_chars =
            static_cast<int>((float)progress / progress_per_tick);
        std::cout << '\r' << leader << open_char;
        for (int i = 0; i != size_x; ++i)
        {
            std::cout << (i < progress_chars ? fill_char : empty_char);
        }
        std::cout << close_char << " (" << (int)progress << "%)";
        (std::cout << ... << printables) << std::flush;
    }

    template <typename... To_Print>
    auto tick(std::string_view leader, To_Print... printables) noexcept -> void
    {
        print(++m_Progress, leader, printables...);
    }

    auto redraw(std::string_view leader) noexcept -> void
    {
        print(m_Progress, leader);
    }

    int m_Progress = -1;
};

class progress_matrix
{
    inline static constexpr std::string_view header = "Progress...";

public:
    progress_matrix(int count) noexcept :
        m_Progress_bars(count)
    {
    }

    template <typename... To_Print>
    auto print(
        int              idx,
        int              progress,
        std::string_view leader = "\0",
        To_Print... printables
    ) noexcept -> void
    {
        std::scoped_lock lock(m_Mutex);
        const auto       n              = (int)m_Progress_bars.size();
        const auto       rows_to_update = n - idx;
        std::cout << '\r';
        for (int i = 1; i != rows_to_update; ++i)
        {
            std::cout << "\x1b[1A"
                      << "\r";
        }

        m_Progress_bars[idx].print(
            progress, leader, std::forward<To_Print>(printables)...
        );

        for (int i = idx + 1; i != n; ++i)
        {
            m_Progress_bars[i].redraw(leader);
            std::cout << '\n';
        }
    }

    template <typename... To_Print>
    auto tick(int idx, std::string_view leader = " ", To_Print... printables)
    {
        std::scoped_lock lock(m_Mutex);
        const auto       n              = (int)m_Progress_bars.size();
        const auto       rows_to_update = n - idx;
        std::cout << '\r';
        for (int i = 1; i != rows_to_update; ++i)
        {
            std::cout << "\x1b[1A"
                      << "\r";
        }

        m_Progress_bars[idx].tick(
            leader, std::forward<To_Print>(printables)...
        );

        for (int i = idx + 1; i != n; ++i)
        {
            m_Progress_bars[i].redraw(leader);
            std::cout << '\n';
        }
    }

private:
    std::vector<progressbar> m_Progress_bars;

    std::mutex m_Mutex;
};

} // namespace progressbar_

#endif // PROGRESSBAR