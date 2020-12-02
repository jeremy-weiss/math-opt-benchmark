from ortools.linear_solver import pywraplp
from numpy.random import random
import numpy as np

SCIP_SOLVER = pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING
GLOP_SOLVER = pywraplp.Solver.GLOP_LINEAR_PROGRAMMING
inf = pywraplp.Solver.Infinity()

customers = 2
facilities = 2
demand = list(map(int, np.array(random(customers)*100 + 1)))
f_cost = list(map(int, np.array(random(facilities)*10000 + 1)))
# capacities = list(map(int, np.array(random(facilities)*300 + 1)))
# capacities = [float('inf')]*facilities
capacities = [999999]*facilities
s_cost = list(map(int, np.array(random(customers*facilities)*100 + 1)))
customers = 2
facilities = 2
demand = [1, 2]
f_cost = [1, 2]
capacities = [9999, 9999]
# s_cost = [2] * (customers*facilities)
s_cost = [1, 1, 2, 2]

def print_solution(xs, ys):
    for j in range(facilities):
        print(f'y[{j}] = {ys[j].solution_value()}')
    for i in range(customers):
        for j in range(facilities):
            print(f'x[{i}][{j}] = {xs[i][j].solution_value()}')


def base():
    solver = pywraplp.Solver('CFL', SCIP_SOLVER)
    # solver.EnableOutput()

    ys = [solver.Var(0, 1, True, f'y{i}') for i in range(facilities)]
    xs = [[solver.Var(0, 1, False, f'x{i}{j}') for j in range(facilities)] for i in range(customers)]

    for i in range(customers):
        solver.Add(sum(xs[i]) == 1)
        for j in range(facilities):
            solver.Add(xs[i][j] <= ys[j])

    for j in range(facilities):
        solver.Add(sum(demand[i] * xs[i][j] for i in range(customers)) <= capacities[j] * ys[j])

    solver.Minimize(sum(f_cost[j] * ys[j] for j in range(facilities)) +
                    sum(sum(demand[i] * s_cost[i*facilities + j] * xs[i][j] for j in range(facilities)) for i in range(customers)))

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        # print_solution(xs, ys)
        print(solver.Objective().Value())
        return solver.Objective().Value()
    else:
        print("Infeasible")


def main():
    for i in range(customers):
        for j in range(facilities):
            s_cost[i * facilities + j] *= demand[j]

    solver = pywraplp.Solver('CFL - Benders', SCIP_SOLVER)
    # solver.EnableOutput()

    ys = [solver.Var(0, 1, True, f'y{j}') for j in range(facilities)]

    # for j in range(facilities):
        # solver.Add(sum(demand) <= sum(list(capacities[j] * ys[j] for j in range(facilities))))
    solver.Add(sum(ys) >= 1)

    ws = [solver.Var(0, inf, False, f'w{j}') for j in range(customers)]
    solver.Minimize(sum(f_cost[j] * ys[j] for j in range(facilities)) + sum(ws))
    solver.Solve()
    print(solver.Objective().Value())

    prev_obj = solver.Objective().Value()
    solution_vals = [y.solution_value() for y in ys]
    while True:
        for j in range(customers):
            costs = [s_cost[i*facilities + j] for i in range(facilities)]
            costs, solution_vals = zip(*sorted(list(zip(costs, solution_vals)), key=lambda x: x[0]))
            partial_sum = 0
            k = 0
            for k in range(facilities):
                partial_sum += solution_vals[k]
                if partial_sum >= 1:
                    break
            # wj *= demand[j]
            solver.Add(ws[j] >= costs[k] - sum(ys[i]*(costs[k]*costs[i] - costs[i]) for i in range(k - 1)))
        status = solver.Solve()
        assert status == solver.OPTIMAL
        print(solver.Objective().Value())
        if prev_obj == solver.Objective().Value():
            break
        else:
            prev_obj = solver.Objective().Value()
        solution_vals = [y.solution_value() for y in ys]

    for y in ys:
        y.SetInteger(True)
    solver.Solve()
    print(solver.Objective().Value())

    prev_obj = solver.Objective().Value()
    solution_vals = [y.solution_value() for y in ys]
    while True:
        for j in range(customers):
            costs = [s_cost[i * facilities + j] for i in range(facilities)]
            costs, solution_vals = zip(*sorted(list(zip(costs, solution_vals)), key=lambda x: x[0]))
            partial_sum = 0
            k = 0
            for k in range(facilities):
                partial_sum += solution_vals[k]
                if partial_sum >= 1:
                    break
            # wj *= demand[j]
            solver.Add(ws[j] >= costs[k] - sum(ys[i] * (costs[k]*costs[i] - costs[i]) for i in range(k - 1)))
        status = solver.Solve()
        assert status == solver.OPTIMAL
        print(solver.Objective().Value())
        if prev_obj == solver.Objective().Value():
            break
        else:
            prev_obj = solver.Objective().Value()
        solution_vals = [y.solution_value() for y in ys]

    # print(solver.Objective().Value())
    return solver.Objective().Value()

if __name__ == '__main__':
    v1 = base()
    v2 = main()
    print(v2)
    assert abs(v2 - v1) <= 0.0001
