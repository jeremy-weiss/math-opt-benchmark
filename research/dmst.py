import tree
from itertools import combinations
import numpy as np
import graph as gph
from ortools.linear_solver import pywraplp

MAX_DEGREE = 3
MAX_ITR = 1000

def dmst_constraints(solver, edge_map, edges, n, weights):
    x, y = np.hsplit(edges, 2)
    x = x.reshape(len(edges))
    y = y.reshape(len(edges))
    solver.Add(sum(edge_map[x, y]) == n - 1)
    for v in range(n):
        xs = [v]*(n-1)
        ys = list(i for i in range(n) if i != v)
        solver.Add(sum(edge_map[xs, ys]) >= 1)
    # Weight is sum of the vertices
    solver.Minimize(sum(weights * edge_map[x, y]))


def check_vertices(graph):
    invalid = []
    for vertex in range(len(graph)):
        if len(graph[vertex]) > MAX_DEGREE:
            invalid.append(vertex)

    return invalid


def vertex_constraints(solver, edge_map, vertices, n):
    for vertex in vertices:
        try:
            xs = list(vertex for i in range(n-1))
            ys = list(other for other in range(n) if other != vertex)
            solver.Add(sum(edge_map[xs, ys]) <= MAX_DEGREE)
        except:
            print("HUH")


def solve(solver, edge_map, edges, n):
    status = solver.Solve()
    # print(tree.enum_names[status])
    assert status == pywraplp.Solver.OPTIMAL
    graph = gph.construct_graph(n, edge_map, edges)
    invalid = check_vertices(graph)
    return invalid, graph


def main():
    np.random.seed(seed=12345)

    n = 150
    edges = np.array(list(combinations(range(n), 2)))
    weights = np.random.random(size=len(edges))
    solver, edge_map = tree.create_solver(edges, True)
    # tree.add_constraints(solver, edge_map, edges, n, weights)
    dmst_constraints(solver, edge_map, edges, n, weights)
    # vertices, graph = solve(solver, edge_map, edges, n)
    cycles, graph = tree.solve(solver, edge_map, edges, n)
    vertices = check_vertices(graph)
    itr = 0
    separated = True
    prev_cycles = None
    # while separated:
        # v_set = set(vertices)
    while itr < MAX_ITR and (vertices or (cycles and frozenset(map(frozenset, cycles)) != prev_cycles)):
        prev_cycles = frozenset(map(frozenset, cycles))
        # for cycle in cycles:
        #     assert frozenset(cycle) not in cycle_set
        #     cycle_set.add(frozenset(cycle))
        tree.augment_constraints(solver, edge_map, cycles)
        vertex_constraints(solver, edge_map, vertices, n)
        cycles, graph = tree.solve(solver, edge_map, edges, n)
        # for vertex in vertices:
        #     assert vertex not in v_set
        #     v_set.add(vertex)
        vertices = check_vertices(graph)
        # verify_cycles(cycles, edge_map)
        itr += 1
        print("Itr", itr, end='\r', flush=True)
        # print(itr, cycles)
        # separated = gph.separate(graph, edge_map)
        # if separated:
        #     print("separated")
        #     cycles = separated
        # print(f"separation oracle: {len(separated)}")


    # print(solver.ExportModelAsLpFormat(False))
    # print(solver.Objective().Value())
    for (i, j) in edges:
        if not gph.close(edge_map[i, j].solution_value(), 0, 0.0001):
            assert gph.close(edge_map[i, j].solution_value(), 1, 0.0001)
            # print(edge_map[i, j], edge_map[i, j].solution_value())
    print("\n")
    gph.viz(edge_map, edges, n)




if __name__ == '__main__':
    main()
