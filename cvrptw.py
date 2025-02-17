import os
import sys
import math
import random
from qubots.base_problem import BaseProblem

def read_elem(filename):

    # Resolve relative path with respect to this module’s directory.
    if not os.path.isabs(filename):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(base_dir, filename)
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]

def compute_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def compute_distance_matrix(customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_matrix = [[None for _ in range(nb_customers)] for _ in range(nb_customers)]
    for i in range(nb_customers):
        distance_matrix[i][i] = 0
        for j in range(nb_customers):
            dist = compute_dist(customers_x[i], customers_y[i], customers_x[j], customers_y[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    return distance_matrix

def compute_distance_depots(depot_x, depot_y, customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_depots = [None] * nb_customers
    for i in range(nb_customers):
        distance_depots[i] = compute_dist(depot_x, customers_x[i], depot_y, customers_y[i])
    return distance_depots

def read_input_cvrptw(filename):
    """
    Reads a CVRPTW instance in the Solomon format.
    Expected tokens:
      - (Skip 4 tokens)
      - nb_trucks, truck_capacity
      - (Skip 13 tokens)
      - depot_x, depot_y
      - (Skip 2 tokens)
      - max_horizon
      - (Skip 1 token)
      - Then, for each customer: 
            customer_id, x, y, demand, ready_time, due_time, service_time.
         Note: due_time in the input represents the latest start time; here we add service_time.
    Returns:
      nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_depots,
      demands, service_time, earliest_start, latest_end, max_horizon
    """
    file_it = iter(read_elem(filename))
    # Skip 4 tokens
    for _ in range(4):
        next(file_it)
    nb_trucks = int(next(file_it))
    truck_capacity = int(next(file_it))
    # Skip next 13 tokens
    for _ in range(13):
        next(file_it)
    depot_x = int(next(file_it))
    depot_y = int(next(file_it))
    for _ in range(2):
        next(file_it)
    max_horizon = int(next(file_it))
    next(file_it)
    customers_x = []
    customers_y = []
    demands = []
    earliest_start = []
    latest_end = []
    service_time = []
    while True:
        val = next(file_it, None)
        if val is None:
            break
        # Customer indices in file are 1-indexed; we subtract 1.
        i = int(val) - 1
        customers_x.append(int(next(file_it)))
        customers_y.append(int(next(file_it)))
        demands.append(int(next(file_it)))
        ready = int(next(file_it))
        due = int(next(file_it))
        stime = int(next(file_it))
        earliest_start.append(ready)
        latest_end.append(due + stime)
        service_time.append(stime)
    nb_customers = i + 1
    distance_matrix = compute_distance_matrix(customers_x, customers_y)
    distance_depots = compute_distance_depots(depot_x, depot_y, customers_x, customers_y)
    return nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_depots, \
           demands, service_time, earliest_start, latest_end, max_horizon

class CVRPTWProblem(BaseProblem):
    """
    Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)

    A fleet of vehicles with uniform capacity must serve a set of customers with known demand,
    service times, and time windows (earliest start and latest end). All vehicles start and end
    at a common depot. The objective is to minimize, in lexicographic order:
      1. Total lateness: For each route, if the end time (including travel from depot to first customer,
         between customers, and back to depot) exceeds a given maximum horizon or if any customer is
         served after its time window, a penalty (lateness) is incurred.
      2. Number of trucks used (i.e., routes that are nonempty).
      3. Total distance traveled.
      
    Candidate Solution:
      A dictionary with key "customersSequences" mapping to a list of lists.
      Each inner list represents the sequence (order) of customer indices (0-indexed) visited by a vehicle.
      The collection of these lists must form a partition of the set {0, ..., nb_customers-1}.

    The evaluation computes for each truck:
      - The total demand (must be ≤ truck_capacity).
      - The route distance (including depot-to-first and last-to-depot legs).
      - The end time of each visit, computed recursively using:
            end_time[0] = max( earliest[start], dist_depot[start] ) + service_time[start]
            end_time[i] = max( earliest[current], end_time[i-1] + dist_matrix[previous, current] ) + service_time[current]
      - Home lateness: max(0, end_time[last] + dist_depot[last] - max_horizon)
      - Customer lateness: sum over customers of max(0, end_time[i] - latest[current])
      
    The overall objective is computed as:
       overall = total_lateness * M^2 + nb_trucks_used * M + total_distance,
    with a large constant M to enforce lexicographic order.
    """
    def __init__(self, instance_file=None, nb_customers=None, nb_trucks=None, truck_capacity=None,
                 dist_matrix=None, dist_depot=None, demands=None, service_time=None,
                 earliest=None, latest=None, max_horizon=None):
        if instance_file is not None:
            self._load_instance(instance_file)
        else:
            if None in (nb_customers, nb_trucks, truck_capacity, dist_matrix, dist_depot,
                        demands, service_time, earliest, latest, max_horizon):
                raise ValueError("Either instance_file or all instance parameters must be provided.")
            self.nb_customers = nb_customers
            self.nb_trucks = nb_trucks
            self.truck_capacity = truck_capacity
            self.dist_matrix = dist_matrix
            self.dist_depot = dist_depot
            self.demands = demands
            self.service_time = service_time
            self.earliest = earliest
            self.latest = latest
            self.max_horizon = max_horizon

    def _load_instance(self, filename):
        (self.nb_customers, self.nb_trucks, self.truck_capacity, self.dist_matrix,
         self.dist_depot, self.demands, self.service_time, self.earliest,
         self.latest, self.max_horizon) = read_input_cvrptw(filename)

    def evaluate_solution(self, solution) -> float:
        PENALTY = 1e9
        M = 1e5  # large constant for lexicographic ordering
        # Check candidate structure.
        if not isinstance(solution, dict) or "customersSequences" not in solution:
            return PENALTY
        routes = solution["customersSequences"]
        if not isinstance(routes, list) or len(routes) != self.nb_trucks:
            return PENALTY

        # Ensure that each customer is visited exactly once.
        visited = [False] * self.nb_customers
        for route in routes:
            if not isinstance(route, list):
                return PENALTY
            for cust in route:
                if not isinstance(cust, int) or cust < 0 or cust >= self.nb_customers:
                    return PENALTY
                if visited[cust]:
                    return PENALTY
                visited[cust] = True
        if not all(visited):
            return PENALTY

        total_lateness = 0
        total_distance = 0
        trucks_used = 0
        for route in routes:
            if len(route) == 0:
                continue
            trucks_used += 1
            # Capacity constraint: total demand for the route.
            route_demand = sum(self.demands[cust] for cust in route)
            if route_demand > self.truck_capacity:
                return PENALTY

            # Compute route distance.
            d_route = self.dist_depot[route[0]] + self.dist_depot[route[-1]]
            for i in range(1, len(route)):
                d_route += self.dist_matrix[route[i-1]][route[i]]
            total_distance += d_route

            # Compute end times for the route.
            end_times = []
            # For first customer:
            et0 = max(self.earliest[route[0]], self.dist_depot[route[0]]) + self.service_time[route[0]]
            end_times.append(et0)
            for i in range(1, len(route)):
                prev_end = end_times[-1]
                travel = self.dist_matrix[route[i-1]][route[i]]
                et = max(self.earliest[route[i]], prev_end + travel) + self.service_time[route[i]]
                end_times.append(et)
            # Home lateness: time after returning to depot.
            home_lat = max(0, end_times[-1] + self.dist_depot[route[-1]] - self.max_horizon)
            lateness_route = home_lat
            for i, cust in enumerate(route):
                lateness_route += max(0, end_times[i] - self.latest[cust])
            total_lateness += lateness_route

        overall = total_lateness * (M**2) + trucks_used * M + total_distance
        return overall

    def random_solution(self):
        # Generate a random permutation partitioning all customers among trucks.
        customers = list(range(self.nb_customers))
        random.shuffle(customers)
        routes = [[] for _ in range(self.nb_trucks)]
        for i, cust in enumerate(customers):
            routes[i % self.nb_trucks].append(cust)
        for route in routes:
            random.shuffle(route)
        return {"customersSequences": routes}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python cvrptw.py input_file [output_file] [time_limit]")
        sys.exit(1)
    instance_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    str_time_limit = sys.argv[3] if len(sys.argv) > 3 else "20"
    # For demonstration: load instance, generate and evaluate a random solution.
    problem = CVRPTWProblem(instance_file=instance_file)
    sol = problem.random_solution()
    obj = problem.evaluate_solution(sol)
    print("Random solution objective:", obj)
