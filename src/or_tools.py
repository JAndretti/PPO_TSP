"""Simple Travelling Salesperson Problem (TSP) on a circuit board."""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


class OR_TOOLS:

    def __init__(self, locations, distance_matrix, or_tools_time) -> None:
        self.locations = locations
        self.distance_matrix = distance_matrix
        self.or_tools_time = or_tools_time
        self.data = self.create_data_model()

    def create_data_model(self):
        """Stores the data for the problem."""
        data = {}
        # Locations in block units
        data["locations"] = self.locations
        data["num_vehicles"] = 1
        data["depot"] = 0
        return data

    def distance_solution(self, solution):
        """Returns the distance of the solution."""
        distance = 0
        for i in range(len(solution) - 1):
            distance += self.distance_matrix[solution[i]][solution[i + 1]]
        return distance

    def get_solution_nodes(self, manager, routing, solution):
        """Returns a list of nodes in the solution."""
        nodes = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            nodes.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        nodes.append(manager.IndexToNode(index))  # Ajouter le dernier nœud
        return nodes

    def main(self):
        """Entry point of the program."""
        # Instantiate the data problem.
        data = self.create_data_model()

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["locations"]), data["num_vehicles"], data["depot"]
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        distance_matrix = self.distance_matrix

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(self.or_tools_time)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            solution = self.get_solution_nodes(manager, routing, solution)
            distance = self.distance_solution(solution)
            return solution, distance
