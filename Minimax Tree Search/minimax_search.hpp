#ifndef MINIMAX_TREE_SEARCH
#define MINIMAX_TREE_SEARCH

#include "Random.h"
#include "parallel_hashmap/phmap.h"

namespace minimax_tree_search
{
template<typename Game_Satate, typename Result>
  class search_state
  {
      using container_type         = phmap::parallel_flat_hash_map;
      using search_state_container = container_type<Game_Satate, Result>;
  private:
      search_state_container m_Search_state{};
    // include an initial size as a ctor parameter
  };
}


#endif // MINIMAX_TREE_SEARCH
