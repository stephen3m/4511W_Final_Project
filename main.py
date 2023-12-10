# import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
import math
import heapq

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
# def astar(graph, start, goal):
    # open_set = PriorityQueue()
    # open_set.put((0, start))
    # came_from = {}
    # g_score = {node: float('inf') for node in graph.nodes()}
    # g_score[start] = 0
    # f_score = {node: float('inf') for node in graph.nodes()}
    # f_score[start] = heuristic(start, goal)

    # while not open_set.empty():
    #     _, current = open_set.get()
    #     if current == goal:
    #         return reconstruct_path(came_from, current)

    #     for neighbor in graph.neighbors(current):
    #         tentative_g_score = g_score[current] + distance(graph, current, neighbor)
    #         if tentative_g_score < g_score[neighbor]:
    #             came_from[neighbor] = current
    #             g_score[neighbor] = tentative_g_score
    #             f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
    #             open_set.put((f_score[neighbor], neighbor))

    # return None

# Greedy Best First Search:

# Djikstra's Algorithm:

# Breadth-First-Search

# Depth-First-Search

######### Heuristics
#### Admissible
def manhattan_distance(node, goal):
    x1, y1 = node[0], node[1]
    x2, y2 = goal[0], goal[1]

    return (x1-x2)+(y1-y2)

def euclidean_distance(node, goal):
    x1, y1 = node[0], node[1]
    x2, y2 = goal[0], node[1]

    return ((x1-x2)**2+(y1-y2)**2)**0.5

#### Inadmissible
def diagonal_distance(node, goal):
    x1, y1 = node[0], node[1]
    x2, y2 = goal[0], node[1]

    return max(abs(x1-x2), abs(y1-y2))

def weighted_manhattan(node, goal):
    return 1.2 * manhattan_distance(node, goal)


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.insert(0, current)
    return path


def main():

    coordinates = parse_file_data()
    G = create_graph(coordinates, 3)  # 3 nearest neighbors

    connect_components(G, coordinates)
    connect_isolated_nodes(G, coordinates)
    # If you want to plot the graph

    pos = {i: coordinates[i] for i in range(len(coordinates))}
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.show()
    return 0

main()