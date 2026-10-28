#ifndef PDP_MEMETIC_H
#define PDP_MEMETIC_H
#include "pdp_search.h"
#include "pdp_tabu_main.h"
LSPResult memeticSearch(const PDPData& data, const std::vector<SolutionEncoding>& initial,
                       SearchContext& context, TSConfig config, int seed);
#endif
