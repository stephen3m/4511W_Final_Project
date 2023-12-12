# import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import time
import math
import heapq
import random

from algorithms import *

# Parse node data from a text file (e.g., coordinates and connections)
def parseTxt():
    file_name = input("Enter the name of the data file: ")
    fp = open("./mapTxt/"+file_name, 'r')

    num_datapoints = int(fp.readline())
    
    coordinates = []
    for i in range(num_datapoints):
        num, x, y = fp.readline().split(" ")
        coordinates.append([float(x),float(y)])

    return coordinates

# Centered data is 0 indexed 
# def center_data(coordinates):
    # x_sum = 0
    # y_sum = 0

    # for x, y in coordinates:
    #     x_sum += x
    #     y_sum += y
    
    # x_mean = x_sum/len(coordinates)
    # y_mean = y_sum/len(coordinates)
    # centered_coordinates = []
    # for i in range(len(coordinates)):
    #     new_x = coordinates[i][0] - x_mean
    #     new_y = coordinates[i][1] - y_mean
    #     centered_coordinates.append([new_x, new_y])

    # return centered_coordinates

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

def drawGraph(coordinates, n):
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
def get_coordinates(G):
    coordinates = {}
    for node in G.nodes(data=True):
        id, node_data = node
        coordinates[id] = node_data['pos']
    return coordinates

def runAlgo(algorithm, G, start, goal, heuristic):
    if heuristic:
        time_start = time.perf_counter()
        path, cost = algorithm(G, start, goal, heuristic)
        time_end = time.perf_counter()
        plot_path(get_coordinates(G), path)
        print(start, " to ", goal)
        print("Runtime: ", 1000*abs(time_end-time_start), "ms")
        print("Path: ", path)
        print("Total distance ", cost)
    else:
        time_start = time.perf_counter()
        path, cost = algorithm(G, start, goal)
        time_end = time.perf_counter()
        plot_path(get_coordinates(G), path)
        print(start, " to ", goal)
        print("Runtime: ", 1000*abs(time_end-time_start), "ms")
        print("Path: ", path)
        print("Total distance ", cost)
    print()
    return

def runAnalysis(G, start, goal):
    print("A* with manhattan distance heuristic: ")
    runAlgo(astar, G, start, goal, manhattan_distance)

    # print("A* with euclidean distance heuristic: ")
    # runAlgo(astar, G, start, goal, euclidean_distance)

    # print("A* with diagonal distance heuristic: ")
    # runAlgo(astar, G, start, goal, diagonal_distance)

    # print("A* with weighted manhattan distance heuristic: ")
    # runAlgo(astar, G, start, goal, weighted_manhattan)


    # print("Greedy with manhattan distance heuristic: ")
    # runAlgo(greedy_best_first_search, G, start, goal, manhattan_distance)

    # print("Greedy with euclidean distance heuristic: ")
    # runAlgo(greedy_best_first_search, G, start, goal, euclidean_distance)

    # print("Greedy with diagonal distance heuristic: ")
    # runAlgo(greedy_best_first_search, G, start, goal, diagonal_distance)

    # print("Greedy with weighted manhattan distance heuristic: ")
    # runAlgo(greedy_best_first_search, G, start, goal, weighted_manhattan)


    # print("Djikstra's (A* no heuristic): ")
    # runAlgo(dijkstra, G, start, goal, None)

    # print("BFS: ")
    # runAlgo(breadth_first_search, G, start, goal, None)

    # print("DFS: ")
    # runAlgo(depth_first_search, G, start, goal, None)

def main():

    coordinates = parseTxt()
    G = drawGraph(coordinates, 4)  # 3 nearest neighbors
    
    connect_components(G, coordinates)
    connect_isolated_nodes(G, coordinates)

    num_cities = len(coordinates)
    print("There are", num_cities, "cities on the graph, pick a start and goal city number from 0 to", num_cities-1)

    start = int(input("Enter the starting city: "))
    if not (0 <= start < num_cities):
        print("Enter a start city within the range!")
        return
    
    goal = int(input("Enter the goal city: "))
    if not (0 <= goal < num_cities):
        print("Enter a goal city within the range!")
        return
    print()
    runAnalysis(G, start, goal)



    return 0

main()