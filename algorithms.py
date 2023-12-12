import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import time
import math
import heapq
import random


def astar(graph, start, goal, heuristic):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph.nodes()}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph.nodes()}
    f_score[start] = heuristic(graph, start, goal)

    open_set_hash = {start}  # To track the items currently in the open set

    while open_set:
        _, current = heapq.heappop(open_set)
        open_set_hash.remove(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            path_cost = g_score[goal]
            return (path, path_cost)

        for neighbor in graph.neighbors(current):
            tentative_g_score = g_score[current] + graph[current][neighbor]['weight']
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(graph, neighbor, goal)
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)

    return None  # Path not found

def reconstruct_path(came_from, current):
    """
    Reconstructs the path from start to goal node as determined by A* algorithm.
    
    :param came_from: A dictionary mapping each node to the node it came from.
    :param current: The current node (goal node at the end of A* algorithm).
    :return: A list representing the path from the start node to the goal node.
    """
    path = [current]
    while current in came_from and came_from[current] is not None:
        current = came_from[current]
        path.insert(0, current)
    return path

# Greedy Best First Search:
def greedy_best_first_search(graph, start, goal, heuristic):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {start: 0}  # Track the cost to reach each node
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current), cost_so_far[current]

        if current not in visited:
            visited.add(current)

            for neighbor in graph.neighbors(current):
                new_cost = cost_so_far[current] + graph[current][neighbor]['weight']
                if neighbor not in visited or new_cost < cost_so_far.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    cost_so_far[neighbor] = new_cost
                    heuristic_cost = heuristic(graph, neighbor, goal)
                    heapq.heappush(open_set, (heuristic_cost, neighbor))

    return None, float('inf')  # Path not found

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
    queue = deque([(start, 0)])  # Store (node, path_cost) tuples
    came_from = {start: None}
    path_cost = {start: 0}

    while queue:
        current, current_cost = queue.popleft()
        if current == goal:
            return reconstruct_path(came_from, current), path_cost[current]

        for neighbor in graph.neighbors(current):
            new_cost = current_cost + graph[current][neighbor]['weight']
            if neighbor not in came_from:
                queue.append((neighbor, new_cost))
                came_from[neighbor] = current
                path_cost[neighbor] = new_cost

    return None, float('inf')  # Path not found

# Depth-First-Search
def depth_first_search(graph, start, goal):
    stack = [(start, 0)]  # Store (node, path_cost) tuples
    came_from = {start: None}
    path_cost = {start: 0}

    while stack:
        current, current_cost = stack.pop()
        if current == goal:
            return reconstruct_path(came_from, current), path_cost[current]

        for neighbor in graph.neighbors(current):
            new_cost = current_cost + graph[current][neighbor]['weight']
            if neighbor not in came_from:
                stack.append((neighbor, new_cost))
                came_from[neighbor] = current
                path_cost[neighbor] = new_cost

    return None, float('inf')  # Path not found

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