#ifndef PDP_TYPES_H
#define PDP_TYPES_H

#include <vector>
#include <string>
#include <utility>
#include <stdexcept>
#include <limits>

using namespace std;

// ============================================================
// === TRUCK--DRONE RESUPPLY ENCODING (per MD specs) ==========
// ============================================================

/**
 * @brief A stop in an encoded truck route.
 *
 * Encoding rules (matches ga_encoding_init_solution.md):
 *  - Depot loading: 0[P]  => node = 0, packages = P
 *  - Route end:     0     => node = 0, packages = {}
 *  - Customer visit: i    => node = i (>0), packages = {}
 *  - Drone resupply at customer: i[P] => node = i (>0), packages = P
 *    The resupply is synchronized at i. If i can already be served by the
 *    truck, service happens first; if i needs a package in P, resupply happens
 *    first.
 *  - Every C1 package loaded at a depot or supplied by a drone must be
 *    delivered before the truck's next depot visit. C2 pickup loads may cross
 *    intermediate depot visits, provided pickup precedes delivery.
 */
struct RouteStop {
    int node = 0;                 // 0 = depot, >0 = customer
    vector<int> packages;         // depot-load packages if node==0; drone-resupply packages if node>0
    int fixed_drone_id = -1;      // Optional MILP validation metadata
    int fixed_drone_trip = -1;    // 1-based sortie order for fixed_drone_id
    double fixed_drone_depart = -1.0;
    double fixed_drone_arrive = -1.0;
    double fixed_drone_return = -1.0;

    RouteStop() = default;
    explicit RouteStop(int n) : node(n) {}
    RouteStop(int n, vector<int> p) : node(n), packages(std::move(p)) {}
};

using Route = std::vector<RouteStop>;
using SolutionEncoding = std::vector<Route>;  // size = numTrucks

// ============================================================
// === NEW: VRP-layer types for TS-ALNS-LSP architecture ======
// ============================================================

// VRP Solution: ONLY customer visit sequences, no loading info.
// vrp[truck_id][position] = customer_node_id (1-indexed)
using VRPSolution = std::vector<std::vector<int>>;

// Lower bound info for a single truck route (paper Proposition 2)
struct LBInfo {
    std::vector<double> li;     // li[k] = LB on departure time at position k
    double lb_completion = 0.0; // LB on completion time (return to depot)
};

// Result from the Loading Subproblem (Layer 2)
struct LSPResult {
    bool   solved    = false;  // false if cutoff triggered or infeasible
    bool   feasible  = false;
    double objective = std::numeric_limits<double>::infinity(); // C_max
    SolutionEncoding encoding; // full encoding if solved
};

/**
 * @brief Exception thrown khi vi phạm ràng buộc tải trọng nghiêm trọng.
 * Được ném ra trong decode_sequence() khi current_load > M_T.
 */
struct InfeasibleException : public std::runtime_error {
    explicit InfeasibleException(const std::string& msg)
        : std::runtime_error("[INFEASIBLE] " + msg) {}
};

/**
 * @brief Type alias cho SeqStop dùng trong ngữ cảnh chromosome của GA.
 *
 * Hai tên phân biệt ngữ cảnh sử dụng:
 *   - SeqStop    : API giải mã lộ trình  (decode_sequence, print_decoded_routes)
 *   - Gene       : API quần thể GA       (pdp_init, pdp_ga, crossover, mutation)
 *
 * Quy ước gene đặc biệt (node_id):
 *   node_id  > 0  →  điểm khách hàng bình thường
 *   node_id == 0  →  vách ngăn chia Truck 1 / Truck 2
 *   node_id == -1 →  lệnh quay về Depot
 */
// NOTE: GA individual is the full encoded solution (SolutionEncoding), not a flat chromosome.

/**
 * @brief Core data structure encapsulating the Pickup-Delivery Problem (PDP) instance.
 * 
 * This structure stores:
 * - Network topology: nodes, customers, depot location
 * - Vehicle fleet: trucks and drones with their specifications
 * - Distance matrices: Manhattan (trucks) and Euclidean (drones)
 * - Physical constraints: capacities, speeds, endurance limits
 */
struct PDPData {
    // === INSTANCE TOPOLOGY ===
    int numNodes = 0;                           // Total nodes in the network (depot + customers)
    int numCustomers = 0;                       // Number of pickup-delivery customer pairs (P, DL, D nodes)
    int numTrucks = 2;                          // Number of available trucks for ground delivery
    int truckCapacity = 50;                     // Load capacity per truck (kg/units)
    double truckSpeed = 30.0;                   // Truck cruising speed (km/h)
    double truckServiceTime = 3.0;              // Time to process each pickup/delivery location (minutes)
    double depotReceiveTime = 5.0;              // Time for truck to receive resupply at depot (minutes)
    int depotIndex = 0;                         // Central depot index (0-based, typically at origin)
    
    // === NODE ATTRIBUTES ===
    vector<pair<double,double>> coordinates;    // (X, Y) geographic coordinates for each node
    vector<string> nodeTypes;                   // Node classification: "DEPOT", "P" (pickup), "DL" (delivery), "D" (optional)
    vector<int> readyTimes;                     // Ready time constraints for time-window feasibility
    vector<int> pairIds;                        // Pairing information for pickup-delivery constraints

    // === DEPOT CONFIGURATION ===
    pair<double, double> depotCenter = {10.0, 10.0};  // Center depot coordinates (primary resupply point)
    pair<double, double> depotBorder = {0.0, 10.0};   // Border depot coordinates (alternative resupply point)
    pair<double, double> depotOutside = {-10.0, 10.0};  // Outside depot coordinates (peripheral resupply point)
    int depotMode = 0;                                 // Depot selection: 0=center, 1=border, 2=outside

    // === CUSTOMER DEMAND & RELEASE DATES ===
    // demands[i] = q_i (weight/units of package i). Default 1 for customers.
    // readyTimes[i] is used as release date r_i (minutes).
    vector<int> demands;
    
    // === DRONE PARAMETERS ===
    int numDrones = 2;                      // Number of drones
    int droneCapacity = 2;                  // Payload capacity per drone trip (weight/units)
    double droneSpeed = 60.0;               // Drone cruising speed (km/h)
    double droneEndurance = 90.0;           // Maximum flight duration (minutes) for range constraint
    double resupplyTime = 5.0;              // Drone-to-truck handover time at rendezvous node (minutes)
    double depotDroneLoadTime = 5.0;        // Drone loading time at depot (minutes)
    double allowedWait = 999999.0;              // Max truck waiting time for drone at rendezvous (minutes)

    // === DISTANCE MATRICES ===
    vector<vector<double>> truckDistMatrix; // Manhattan distance matrix (truck routing, urban/grid model)
    vector<vector<double>> droneDistMatrix; // Euclidean distance matrix (drone routing, straight-line model)

    // === UTILITY METHODS ===
    
    /**
     * @brief Get the starting index for truck separator nodes in the chromosome encoding.
     * Separators mark points where truck routes begin.
     */
    // Separator / depot-return are not used in the MD encoding.
    
    /**
     * @brief Customer set C = {1..N} in the MD specs.
     * For compatibility with current instance files, we treat every node index > 0
     * as a customer (one package per customer).
     */
    bool isCustomer(int id) const {
        return (id > 0 && id < numNodes);
    }
    
    /**
     * @brief Check if a node represents the central depot.
     * @param id Node identifier
     * @return true if id matches the depot index
     */
    bool isDepot(int id) const {
        return (id == depotIndex);
    }
    
    int getDroneCapacity() const {
        if (numCustomers + 1 <= 20) return 2;
        return 10;
    }
};

struct ResupplyEvent {
    vector<int> customer_ids;      // Packages delivered in this trip
    int resupply_point;            // Rendezvous node h (a customer node)
    int drone_id;                  // Drone ID
    int truck_id;                  // Truck ID
    double drone_depart_time;      // Time when drone departs from depot
    double drone_arrive_time;      // Time when drone arrives at resupply point
    double truck_arrive_time;      // Time when truck arrives at resupply point
    double resupply_start_time;    // Time when resupply starts (max of drone & truck arrival)
    double resupply_end_time;      // Time when resupply ends (after handover)
    double drone_return_time;      // Time when drone returns to depot
    double total_flight_time;      // Total flight duration (for endurance constraint check)
    double truck_delivery_end;     // Time when truck finishes its route
};

// Structure storing detailed information about each truck's route
struct TruckRouteInfo {
    int truck_id;
    vector<int> route;        // Sequence of visited nodes (including depot)
    vector<double> arrival_times;  // Arrival time at each node
    vector<double> departure_times; // Departure time from each node
    double completion_time;   // Route completion time (return to depot)
};

// Structure representing a complete PDP solution
struct PDPSolution {
    // Encoded solution per MD specs
    SolutionEncoding encoded_routes;

    // Decoded/summary routes (customers only, for convenience)
    vector<vector<int>> customer_routes;

    double totalCost = 0.0;     // Objective C_max
    double totalPenalty = 0.0;  // Penalty for constraint violations
    bool isFeasible = false;    // Solution feasibility flag
    
    // Detailed solution information
    vector<TruckRouteInfo> truck_details;
    vector<ResupplyEvent> resupply_events;
    vector<double> drone_completion_times; // Completion time for each drone
    
    // Drone execution order (indices into resupply_events)
    vector<int> drone_order;
};

#endif
