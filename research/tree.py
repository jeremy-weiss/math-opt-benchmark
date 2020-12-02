from ortools.linear_solver import pywraplp
from itertools import chain, combinations
from graph import *
from sys import argv
import dmst

# TODO Time it

CYCLES = True if len(argv) == 1 else False
# CYCLES = False
EXT_SOLVER = pywraplp.Solver.GLOP_LINEAR_PROGRAMMING
EXT_SOLVER = pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING
MAX_ITR = 100000


def edge_combinations(edge_map, subset):
    return [edge_map[i, j] for (i, j) in combinations(subset, 2) if edge_map[i, j]]


def create_solver(edges, integral=False):
    solver = pywraplp.Solver('Minimum Spanning Tree', EXT_SOLVER)
    solver.EnableOutput()
    # edge_map = {(i,j): solver.Var(0, 1, integral, f'X({i}, {j})') for (i, j) in edges}
    n = num_vertices(edges)
    edge_map = np.ndarray((n, n), dtype=object)
    for (i, j) in edges:
        var = solver.Var(0, 1, integral, f'X({i}, {j})')
        edge_map[i][j] = edge_map[j][i] = var
    return solver, edge_map


def add_constraints(solver, edge_map, edges, n, weights):
    x, y = np.hsplit(edges, 2)
    x = x.reshape(len(edges))
    y = y.reshape(len(edges))
    solver.Add(sum(edge_map[x, y]) == n - 1)
    # Weight is sum of the vertices : np.sum(edges, axis=1)
    solver.Minimize(sum(weights * edge_map[x, y]))


def augment_constraints(solver, edge_map, cycles):
    for i in range(min(len(cycles), 100)):
        cycle = cycles[i]
        feasible_edges = list(edge_combinations(edge_map, cycle))
        # assert len(feasible_edges) == len(cycle) * (len(cycle)-1) / 2
        solver.Add(sum(feasible_edges) <= len(cycle) - 1)


enum_names = {value: name for name, value in list(vars(pywraplp.Solver).items())[36:42]}
def solve(solver, edge_map, edges, n):
    status = solver.Solve()
    # print(enum_names[status])
    assert status == pywraplp.Solver.OPTIMAL
    graph = construct_graph(n, edge_map, edges)
    invalid = cycle_vertices(graph) if CYCLES else invalid_components(graph, edge_map)
    return invalid, graph


def main():
    # edges = np.array([
    #     (0, 3),
    #     (0, 2),
    #     (0, 1),
    #     (1, 2),
    #     (2, 3),
    #     (3, 4),
    #     (4, 1),
    #     (4, 2)
    # ])
    # n = 5

    # n = 7  # 0.5 solution value
    # n = 15  # Infeasible
    # n = 20
    np.random.seed(seed=12345)
    for n in range(200, 201):
        for _ in range(1):
            edges = np.array(list(combinations(range(n), 2)))
            weights = np.random.random(len(edges))

            # def weights(edge):
            #     return sum(edge)

            # cycle_set = set()

            solver, edge_map = create_solver(edges)
            # add_constraints(solver, edge_map, edges, n, weights)
            dmst.dmst_constraints(solver, edge_map, edges, n, weights)
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
                    print("Itr", itr, end='\r')
                    # print(itr, cycles)
                separated = separate(graph, edge_map)
                print(f"separation oracle: {len(separated)}")
                cycles = separated

            # verify_mst(solver, edge_map, n)
            print(f"{n}:{_}/10: Correct", end='\r')

        print(solver.ExportModelAsLpFormat(False))
        print(solver.Objective().Value())
        for (i, j) in edges:
            if not close(edge_map[i, j].solution_value(), 0, 0.0001):
                # print(edge_map[i, j], edge_map[i, j].solution_value())
                assert close(edge_map[i, j].solution_value(), 1, 0.0001)

    # viz(edge_map, edges, n)


def verify_mst(solver, edge_map, n):
    def power_set(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(3, len(s) + 1))

    obj = solver.Objective().Value()
    last = -1
    ss = list(power_set(range(n)))
    # print("power set:", len(ss))
    for s in ss:
        if len(s) != last:
            last = len(s)
            # print(f"{len(s)} / {n}")
        es = edge_combinations(edge_map, s)
        solver.Add(sum(es) <= len(es) - 1)

    solver.Solve()
    # print(obj, solver.Objective().Value())
    assert close(obj, solver.Objective().Value(), 0.0001)


def verify_cycles(cycles, edge_map):
    for cycle in cycles:
        v1 = cycle[0]
        for v in cycle[1:]:
            assert close(edge_map[v1, v].solution_value(), 1, 0.0001)
            v1 = v
        assert close(edge_map[v1, cycle[0]].solution_value(), 1, 0.0001)


if __name__ == '__main__':
    main()
