# import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import time
import math
import heapq
import random

# Parse node data from a text file (e.g., coordinates and connections)
def parse_file_data():
    file_name = input("Enter the name of the data file: ")
    fp = open(file_name, 'r')

    num_datapoints = int(fp.readline())
    
    coordinates = []
    for i in range(num_datapoints):
        num, x, y = fp.readline().split(" ")
        coordinates.append([float(x),float(y)])

    return coordinates

# Centered data is 0 indexed 
def center_data(coordinates):
    x_sum = 0
    y_sum = 0

    for x, y in coordinates:
        x_sum += x
        y_sum += y
    
    x_mean = x_sum/len(coordinates)
    y_mean = y_sum/len(coordinates)
    centered_coordinates = []
    for i in range(len(coordinates)):
        new_x = coordinates[i][0] - x_mean
        new_y = coordinates[i][1] - y_mean
        centered_coordinates.append([new_x, new_y])

    return centered_coordinates

def n_nearest_neighbors(cur_city, coordinates, n):
    cur_x = coordinates[cur_city][0]
    cur_y = coordinates[cur_city][1]

    n_nearest = []
    for i in range(len(coordinates)):
        if i == cur_city:
            continue
        x,y = coordinates[i][0], coordinates[i][1]
        dist = math.dist((cur_x, cur_y), (x, y))

        heapq.heappush(n_nearest, (-dist, i))
        if len(n_nearest) > n:
            heapq.heappop(n_nearest)

    return n_nearest

def create_graph(coordinates, n):
    G = nx.Graph()
    num_cities = len(coordinates)

    # Add nodes with position attributes
    for i, coord in enumerate(coordinates):
        G.add_node(i, pos=coord)

    # Add edges based on n nearest neighbors
    nearest_neighbors = {i: [] for i in range(num_cities)}
    for i in range(num_cities):
        nearest_neighbors[i] = n_nearest_neighbors(i, coordinates, n)

    for node, neighbors in nearest_neighbors.items():
        for dist, neighbor in neighbors:
            G.add_edge(node, neighbor, weight=math.dist(coordinates[node], coordinates[neighbor]))
    
    return G

def connect_components(G, coordinates):
    components = list(nx.connected_components(G))
    if len(components) > 1:
        for i in range(len(components) - 1):
            min_dist = float('inf')
            closest_nodes = None
            for node_in_comp in components[i]:
                for node_in_other_comp in components[i + 1]:
                    dist = math.dist(coordinates[node_in_comp], coordinates[node_in_other_comp])
                    if dist < min_dist:
                        min_dist = dist
                        closest_nodes = (node_in_comp, node_in_other_comp)

            if closest_nodes:
                G.add_edge(*closest_nodes, weight=min_dist)

def connect_isolated_nodes(G, coordinates):
    isolated = list(nx.isolates(G))
    for node in isolated:
        min_dist = float('inf')
        closest_node = None
        for potential_neighbor in G.nodes:
            if potential_neighbor != node:
                dist = math.dist(coordinates[node], coordinates[potential_neighbor])
                if dist < min_dist:
                    min_dist = dist
                    closest_node = potential_neighbor

        if closest_node is not None:
            G.add_edge(node, closest_node, weight=min_dist)

# Create a graph and add nodes and edges

# Implement the searching algorithms

# Astar:
def astar(graph, start, goal, heuristic):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph.nodes()}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph.nodes()}
    f_score[start] = heuristic(graph, start, goal)

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = reconstruct_path(came_from, current)
            path_cost = g_score[goal]
            return (path, path_cost)

        for neighbor in graph.neighbors(current):
            tentative_g_score = g_score[current] + graph[current][neighbor]['weight']
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(graph, neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # Path not found

def reconstruct_path(came_from, current):
    """
    Reconstructs the path from start to goal node as determined by A* algorithm.
    
    :param came_from: A dictionary mapping each node to the node it came from.
    :param current: The current node (goal node at the end of A* algorithm).
    :return: A list representing the path from the start node to the goal node.
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.insert(0, current)
    return path

# Greedy Best First Search:
def greedy_best_first_search(graph, start, goal, heuristic):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)

        if current not in visited:
            visited.add(current)

            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    came_from[neighbor] = current
                    heuristic_cost = heuristic(graph, neighbor, goal)
                    heapq.heappush(open_set, (heuristic_cost, neighbor))

    return None  # Path not found if the loop ends

# Djikstra's Algorithm:
def dijkstra(graph, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph.nodes()}
    g_score[start] = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = reconstruct_path(came_from, current)
            path_cost = g_score[goal]  # Total cost of the path
            return path, path_cost

        for neighbor in graph.neighbors(current):
            tentative_g_score = g_score[current] + graph[current][neighbor]['weight']
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                heapq.heappush(open_set, (g_score[neighbor], neighbor))

    return None, float('inf')  # Path not found

# Breadth-First-Search
def breadth_first_search(graph, start, goal):
    queue = deque([start])
    came_from = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in graph.neighbors(current):
            if neighbor not in came_from:
                queue.append(neighbor)
                came_from[neighbor] = current

    return None  # Path not found

# Depth-First-Search
def depth_first_search(graph, start, goal):
    stack = [start]
    came_from = {start: None}

    while stack:
        current = stack.pop()
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in graph.neighbors(current):
            if neighbor not in came_from:
                stack.append(neighbor)
                came_from[neighbor] = current

    return None  # Path not found

######### Heuristics
#### Admissible
def manhattan_distance(graph, node, goal):
    x1, y1 = graph.nodes[node]['pos']
    x2, y2 = graph.nodes[goal]['pos']
    return abs(x1 - x2) + abs(y1 - y2)

def euclidean_distance(graph, node, goal):
    x1, y1 = graph.nodes[node]['pos']
    x2, y2 = graph.nodes[goal]['pos']
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

#### Inadmissible
def diagonal_distance(graph, node, goal):
    x1, y1 = graph.nodes[node]['pos']
    x2, y2 = graph.nodes[goal]['pos']
    return max(abs(x1 - x2), abs(y1 - y2))

def weighted_manhattan(graph, node, goal):
    return 1.2 * manhattan_distance(graph, node, goal)


def plot_path(coordinates, path):
    # Plotting all the points
    for coord in coordinates:
        x = coordinates[coord][0]
        y = coordinates[coord][1]
        plt.scatter(x, y, c='blue')

    # Plotting the path
    for i in range(len(path) - 1):
        point1 = coordinates[path[i]]
        point2 = coordinates[path[i + 1]]
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], c='red')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Path Found by A* Algorithm')
    plt.show()

# Assuming you have a list of coordinates and a path returned by A* algorithm
# coordinates = [...]
# path, _ = astar(G, start, goal, lambda node, goal: euclidean_distance(pos, node, goal))

def test(search_algorithm, cycles, heuristic, G):
    num_cities = len(G)
    if heuristic:
        time_start = time.perf_counter()
        for i in range(cycles):
            start = random.randint(0, num_cities-1)
            goal = start
            while goal == start:
                goal = random.randint(0, num_cities-1)

            search_algorithm(G, start, goal, heuristic)
        time_end = time.perf_counter()
        return abs(time_end-time_start)
    
    else:
        time_start = time.perf_counter()
        for i in range(cycles):
            start = random.randint(0, num_cities-1)
            goal = start
            while goal == start:
                goal = random.randint(0, num_cities-1)

            search_algorithm(G, start, goal)
        time_end = time.perf_counter()
        return abs(time_end-time_start)
    

def main():

    coordinates = parse_file_data()
    G = create_graph(coordinates, 3)  # 3 nearest neighbors

    connect_components(G, coordinates)
    connect_isolated_nodes(G, coordinates)

    pos = {i: coordinates[i] for i in range(len(coordinates))}

    print("A* using admissible heuristics")
    print("Time taken for A* search (manhattan_distance): ", test(astar, 1000, manhattan_distance, G))
    print("Time taken for A* search (euclidean distance): ", test(astar, 1000, euclidean_distance, G))
    print()
    print("A* using inadmissible heursitics")
    print("Time taken for A* search (diagonal_distance): ", test(astar, 1000, euclidean_distance, G))
    print("Time taken for A* search (weighted_manhattan): ", test(astar, 1000, euclidean_distance, G))
    print()
    print("Greedy Best-first-search using admissible heuristics")
    print("Time taken for Greedy search (manhattan_distance): ", test(greedy_best_first_search, 1000, manhattan_distance, G))
    print("Time taken for Greedy search (euclidean distance): ", test(greedy_best_first_search, 1000, euclidean_distance, G))
    print()
    print("Greedy Best-first-search using inadmissible heursitics")
    print("Time taken for Greedy search (diagonal_distance): ", test(greedy_best_first_search, 1000, euclidean_distance, G))
    print("Time taken for Greedy search (weighted_manhattan): ", test(greedy_best_first_search, 1000, euclidean_distance, G))
    print()
    print("Djikstra's (A* No Heuristics)")
    print("Time taken for Djikstra's search (manhattan_distance): ", test(dijkstra, 1000, None, G))
    print()
    print("Breadth-first search")
    print("Time taken for BFS", test(breadth_first_search, 1000, None, G))
    print()
    print("Depth-first search")
    print("Time taken for DFS", test(depth_first_search, 1000, None, G))
    return 0

main()