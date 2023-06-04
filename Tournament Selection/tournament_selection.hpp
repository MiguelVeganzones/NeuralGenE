#include "Random.h"
#include "activation_functions.hpp"
#include <algorithm>
#include <array>

namespace tournament_selction
{

template <typename T>
concept agent_type = requires(T t) { t.clone(); };

template <agent_type Agent_Type>
class tournament
{
};


} // namespace tournament_selction
