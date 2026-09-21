#ifndef PDP_FITNESS_H
#define PDP_FITNESS_H

#include "pdp_types.h"
#include <vector>
#include <string>



/**
 * @brief Decode and evaluate an encoded solution per MD specs.
 *
 * Input encoding:
 *   SolutionEncoding = [Route_1, ..., Route_n]
 *   Each Route_k is a list of RouteStop:
 *     - 0[P]  depot load packages P
 *     - i     serve customer i
 *     - i[P]  drone resupply packages P at i, synchronized with service at i
 *     - 0     end route
 */
PDPSolution decode_solution(const SolutionEncoding& encoded, const PDPData& data,
                            bool throw_on_infeasible = false);

/**
 * @brief In lộ trình chi tiết của 2 xe tải và thứ tự bay của drone ra stdout.
 * Dùng để debug / visualization kết quả decode_sequence().
 *
 * @param sol  Kết quả từ decode_sequence()
 * @param data Thông tin bài toán
 */
void print_decoded_routes(const PDPSolution& sol, const PDPData& data);

#endif // PDP_FITNESS_H
