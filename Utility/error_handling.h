#ifndef ERROR_HANDLING_UTILITY
#define ERROR_HANDLING_UTILITY

#include <cassert>
#include <utility>

[[noreturn]] inline void assert_unreachable()
{
#if !NDEBUG
    assert(false);
#endif
    std::unreachable();
}


#endif