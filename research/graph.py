import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ortools.graph import pywrapgraph
from tree import *

def num_vertices(edges):
    return np.max(np.array(edges).flatten()) + 1


def construct_graph(n, edge_map, edges):
    epsilon = 0.0001
    graph = [[] for _ in range(n)]
    for (i, j) in edges:
        x = edge_map[i, j]
        # Verify that the solution is always integral
        # assert close(x.solution_value(), 0, epsilon) or close(x.solution_value(), 1, epsilon)
        # if not close(x.solution_value(), 0, epsilon):
        if x.solution_value() >= epsilon:
            graph[i].append(j)
            # if CYCLES:
            graph[j].append(i)
    return graph


def construct_graph_from_edges(edges):
    n = num_vertices(edges)
    graph = [[] for _ in range(n)]
    for (i, j) in edges:
        graph[i].append(j)
        graph[j].append(i)
    return graph


def cycle_vertices(graph):
    cycles = []
    node_stack = []
    edge_indices = [0] * len(graph)
    e_used = set()  # TODO remove hashing
    visited = [False] * len(graph)
    num_visited = 0
    while num_visited < len(graph):
        index = visited.index(False)
        if not graph[index]:  # Disconnected vertex
            visited[index] = True
            num_visited += 1
            continue

        node_stack.append(index)
        while node_stack:
            head = node_stack[-1]
            other = graph[head][edge_indices[head]]
            while frozenset([head, other]) in e_used:
                edge_indices[head] += 1
                if edge_indices[head] == len(graph[head]):
                    break
                other = graph[head][edge_indices[head]]
            if edge_indices[head] == len(graph[head]):
                num_visited += 1
                visited[head] = True
                node_stack.pop()
            else:
                e_used.add(frozenset([head, other]))
                if other in node_stack:
                    cycles.append(node_stack[node_stack.index(other):])
                else:
                    node_stack.append(other)

    return cycles


def invalid_components(graph, edge_map):
    visited = [False] * len(graph)
    num_visited = 0
    components = []
    node_stack = []
    while num_visited < len(graph):
        component = []
        index = visited.index(False)
        node_stack.append(index)
        visited[index] = True
        while node_stack:
            head = node_stack.pop()
            component.append(head)
            for other in graph[head]:
                if not visited[other]:
                    node_stack.append(other)
                    visited[other] = True

        components.append(component)
        num_visited += len(component)

    invalid = []
    for component in components:
        # TODO might use graph for edges ONLY if all solution_value are edgesg
        if sum(edge_map[x, y].solution_value() for x in component for y in graph[x] if x in component) / 2 > len(component) - 0.9999:
            invalid.append(component)
        # pairs = combinations(component, 2)
        # if sum(edge_map[x, y].solution_value() for (x, y) in pairs) > len(component) - 0.9999:
        #     invalid.append(component)
    return invalid



def separate(graph, edge_map):
    SCALE = 10000

    #for i1 in range(len(graph)):
    for i1, i2 in combinations(range(len(graph)), 2):
        start_nodes = []
        end_nodes = []
        capacities = []

        def build_flow(s, e, c):
            start_nodes.append(int(s))
            end_nodes.append(int(e))
            capacities.append(int(SCALE * c))

        s, t = len(graph), len(graph) + 1
        for v1 in range(len(graph)):
            for v2 in graph[v1]:
                build_flow(v1, v2, edge_map[v1, v2].solution_value() / 2)
                # if {v1, v2} == {i1, i2}:
                #     capacities[-1] = 2**30

            build_flow(s, v1, sum(edge_map[v1, v2].solution_value() for v2 in graph[v1]) / 2)
            if v1 == i1 or v1 == i2:
                capacities[-1] = 2**30
            build_flow(v1, t, 1)


        max_flow = pywrapgraph.SimpleMaxFlow()
        for i in range(len(start_nodes)):
            max_flow.AddArcWithCapacity(start_nodes[i], end_nodes[i], capacities[i])
        if max_flow.Solve(s, t) == max_flow.OPTIMAL:
            flow = max_flow.OptimalFlow() / SCALE
            # N = 50 was getting flow = 49.9992 AND flow = 50.0016
            # N = 100 ==> flow = 100.0022
            if not close(flow, len(graph), 0.1) and len(max_flow.GetSourceSideMinCut()) > 2:
                cut = [v for v in max_flow.GetSourceSideMinCut() if v != s]
                # print(flow - len(graph))
                # print(cut)
                return [cut]
        else:
            raise Exception('There was an issue with the max flow input.')

    return []


# Copied from stack overflow :)
def viz(edge_map, edges, n):
    A = np.zeros((n, n))
    for (i, j) in edges:
        if edge_map[i, j] and close(edge_map[i, j].solution_value(), 1, 0.0001):
            A[i, j] = A[i, j] = 1
    G = nx.from_numpy_matrix(A)
    nx.draw(G)
    plt.show()


def close(value, target, epsilon):
    return abs(target - value) < epsilon
