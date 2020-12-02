#!/bin/env python

# import graph_tool.all as gt
# import timeit
import re
from tree import *
from graph import *
# import sys

# u = gt.load_graph("pgp_undirected.xml")

if __name__ == "__main__":
    # n=100
    # print(timeit.timeit("gt.min_spanning_tree(u)", setup="from __main__ import gt, u", number=n)/n)
    # tree = ET.parse('pgp_undirected.xml').getroot()
    # nodes = root.findall('./graph/node')
    # print()
    with open('pgp_undirected.xml', 'r') as f:
        text = f.read()
    n = len(re.findall(r'<node', text))
    edges = re.findall(r'<edge id="e\d*" source="n(\d*)" target="n(\d*)">', text)
    edges = np.array(list(map(lambda x: list(map(int, x)), edges)))
    # graph = {i: [] for i in range(n)}
    # for edge in edges:
    #     int_edge = list(map(int, edge))
    #     fs = frozenset(int_edge)
    #     graph[int_edge[0]].append(fs)
    #     graph[int_edge[1]].append(fs)
    #
    weights = np.random.random(len(edges))

    solver, edge_map = create_solver(edges)
    add_constraints(solver, edge_map, edges, n, weights)
    cycles, graph = solve(solver, edge_map, edges, n)
    itr = 0
    separated = True
    prev_cycles = None
    while separated:
        while cycles and itr < MAX_ITR and frozenset(map(frozenset, cycles)) != prev_cycles:
            prev_cycles = frozenset(map(frozenset, cycles))
            # for cycle in cycles:
            #     assert frozenset(cycle) not in cycle_set
            #     cycle_set.add(frozenset(cycle))
            augment_constraints(solver, edge_map, cycles)
            cycles, graph = solve(solver, edge_map, edges, n)
            # verify_cycles(cycles, edge_map)
            itr += 1
            print("Itr", itr)
            # print(itr, cycles)
        separated = separate(graph, edge_map)
        print(f"separation oracle: {len(separated)}")
        cycles = separated

    print(solver.Objective().Value())
    viz(edge_map, edges, n)
