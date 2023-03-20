
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
    static inline void add(const std::string_view sv)
    {
        for (const auto c : sv)
            s_log += c;
        s_log += '\n';
        ++s_logs;
    }

    static inline void clear()
    {
        s_log  = s_Initial_message;
        s_logs = 0;
    }

    static inline void flush_log()
    {
        if (!s_Initialized)
        {
            init();
        }

        if (std::ofstream out(s_Logging_file_path); out.is_open())
        {
            out << s_log << s_Final_message << std::endl;
        }
        else
        {
            std::cout << "Could not log info to file\n";
        }

        std::cout << s_log << s_Final_message << std::endl;
    }

    [[nodiscard]] static inline std::string_view get()
    {
        return s_log.c_str();
    }

    [[nodiscard]] static inline size_t get_num_logs()
    {
        return s_logs;
    }

private:
    static void init()
    {
        if (const auto error_code = ::localtime_r(&s_Compile_time, &s_tm); error_code == 0)
        {
            s_Log_filename << std::put_time(&s_tm, "%Y_%m_%d__%H_%M_%S") << "_Log.txt";
            s_Logging_file_path /= s_Log_filename.str();
            s_Initialized = true;
        }
        else
        {
            std::cerr << "Error while formatting local time. Error code: std::errno(" << error_code << ')';
        }
    }

    static inline bool s_Initialized = false;

    static inline const std::time_t s_Compile_time = std::time(nullptr); // Current time
    static inline std::tm           s_tm;
    static inline std::stringstream s_Log_filename;

    static inline std::filesystem::path s_Logging_file_path = "../Logs";

    static inline const std::string s_Initial_message = "<LOG>\n-----------------------------------------\n\n";
    static inline const std::string s_Final_message   = "\n-----------------------------------------\n</LOG>";

    static inline size_t      s_logs;
    static inline std::string s_log = s_Initial_message;
};

#endif // !LOG_UTILITY
