
/*
Exception safety: https://www.boost.org/community/exception_safety.html

*/

#ifndef LOG_UTILITY
#define LOG_UTILITY

#include <iostream>
#include <string_view>

class log
{
public:
  static inline void add(const std::string_view sv)
  {
    s_log += '\n';
    for (const auto c : sv)
      s_log += c;
    ++s_logs;
  }

  static inline void clear()
  {
    s_log = s_Initial_message;
    s_logs = 0;
  }

  static inline void flush_log()
  {
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
  static inline const std::string s_Initial_message = "<LOG>\n\n";
  static inline const std::string s_Final_message = "</LOG>\n\n";

  static inline size_t      s_logs;
  static inline std::string s_log = s_Initial_message;
};

#endif // !LOG_UTILITY
