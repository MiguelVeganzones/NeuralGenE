
/*
Exception safety: https://www.boost.org/community/exception_safety.html

*/

#ifndef LOG_UTILITY
#define LOG_UTILITY

#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string_view>

class log
{
public:
    inline static void add(const std::string s)
    {
        s_log += s + '\n';
        ++s_logs;
    }

    inline static void clear()
    {
        s_log  = s_Initial_message;
        s_logs = 0;
    }

    inline static void flush_log()
    {
        if (!s_Initialized)
        {
            init();
        }

        if (std::ofstream out(s_Logging_file_path); out.is_open())
        {
            out << s_log << s_Final_message << '\n';
        }
        else
        {
            std::cout << "Could not log info to file\n";
        }

        std::cout << s_log << s_Final_message << '\n';
    }

    [[nodiscard]]
    inline static std::string_view get()
    {
        return s_log.c_str();
    }

    [[nodiscard]]
    inline static std::size_t get_num_logs()
    {
        return s_logs;
    }

private:
    static void init()
    {
        s_Log_filename << std::put_time(&s_tm, "%Y_%m_%d__%H_%M_%S")
                       << "_Log.txt";
        s_Logging_file_path /= s_Log_filename.str();
        s_Initialized = true;
    }

    inline static bool s_Initialized = false;

    inline static const std::time_t s_Compile_time =
        std::time(nullptr); // Current time
    inline static const std::tm     s_tm = *std::localtime(&s_Compile_time);
    inline static std::stringstream s_Log_filename;

    inline static std::filesystem::path s_Logging_file_path = "../Logs";

    inline static const std::string_view s_Initial_message =
        "<LOG>\n-----------------------------------------\n\n";
    inline static const std::string_view s_Final_message =
        "\n-----------------------------------------\n</LOG>";

    inline static std::size_t s_logs;
    inline static std::string s_log = std::string(s_Initial_message);
};

#endif // !LOG_UTILITY
