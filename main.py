# import networkx as nx
import matplotlib.pyplot as plt

# Parse node data from a text file (e.g., coordinates and connections)
def parse_file_data():
    file_name = input("Enter the name of the data file: ")
    fp = open(file_name, 'r')

    num_datapoints = int(fp.readline())
    # Data will be stored inside a dictionary: [key: the city number, value: coordinates tuple]
    coordinates = []
    for i in range(num_datapoints):
        num, x, y = fp.readline().split(" ")
        coordinates.append([float(x),float(y)])

    return coordinates

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

# Create a graph and add nodes and edges

# Implement the searching algorithms

# Astar:
# def astar(graph, start, goal):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph.nodes()}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph.nodes()}
    f_score[start] = heuristic(start, goal)

    while not open_set.empty():
        _, current = open_set.get()
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in graph.neighbors(current):
            tentative_g_score = g_score[current] + distance(graph, current, neighbor)
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                open_set.put((f_score[neighbor], neighbor))

    return None


# Greedy Best First Search:


# Djikstra's Algorithm:


# Breadth-First-Search


# Depth-First-Search

######### Heuristics
#### Admissible
def manhattan_distance(node, goal):
    x1, y1 = node[0], node[1]
    x2, y2 = goal[0], node[1]

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


# Create a Matplotlib figure and axis
# fig, ax = plt.subplots()

# Draw the graph using Matplotlib
# pos = nx.spring_layout(graph)  # Define node positions (you can choose a layout)
# nx.draw(graph, pos, with_labels=True, node_size=300)

# Add labels to nodes (optional)
# labels = {node: f"{node}" for node in graph.nodes()}
# nx.draw_networkx_labels(graph, pos, labels)

# Show the visualization
# plt.show()

# graph = nx.Graph()
# # Add nodes and edges to the graph

# # Call the A* algorithm
# start_node = 'A'  # Replace with your actual start node
# goal_node = 'B'   # Replace with your actual goal node
# path = astar(graph, start_node, goal_node)

# # Visualize the graph and path
# pos = nx.spring_layout(graph)  # Adjust layout as needed
# nx.draw(graph, pos, with_labels=True, node_size=300)
# nx.draw_networkx_labels(graph, pos, labels={node: node for node in graph.nodes()})
# if path:
#     path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
#     nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='r', width=2)
# plt.show()
def main():
    print(parse_file_data())
    print("______________________________________________________________________")
    print(center_data(parse_file_data()))
    return 0

main()