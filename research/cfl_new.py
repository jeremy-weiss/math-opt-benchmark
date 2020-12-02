from ortools.linear_solver import pywraplp
from numpy.random import random
import numpy as np
from time import time

SCIP_SOLVER = pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING
GLOP_SOLVER = pywraplp.Solver.GLOP_LINEAR_PROGRAMMING
inf = pywraplp.Solver.Infinity()

# customers = 5
# facilities = 4
# demand = [1, 2, 3, 4, 5]
# f_cost = [1, 5, 3, 0.1]
# capacities = [9, 20, 9, 1]
# s_cost = 1
customers = 50
facilities = 40
demand = list(map(int, np.array(random(customers)*100 + 1)))
f_cost = list(map(int, np.array(random(facilities)*10000 + 1)))
# f_cost.sort()
capacities = list(map(int, np.array(random(facilities)*30000 + 1)))
# capacities = [float('inf')]*facilities
s_cost = 1


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
                    sum(sum(demand[i] * (s_cost+j) * xs[i][j] for j in range(facilities)) for i in range(customers)))

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        # print_solution(xs, ys)
        print(solver.Objective().Value())
        return solver.Objective().Value()
    else:
        print("Infeasible")


def main():
    global status
    solver = pywraplp.Solver('CFL - Benders', SCIP_SOLVER)
    # solver.EnableOutput()

    ys = [solver.Var(0, 1, False, f'y{i}') for i in range(facilities)]

    # solver.Add(sum(ys) >= 2)

    for j in range(facilities):
        solver.Add(sum(demand) <= sum(list(capacities[j] * ys[j] for j in range(facilities))))
    # solver.Add(sum(ys) >= 1)
    # ws = [solver.Var(0, inf, False, f'w{j}') for j in range(customers)]
    w = solver.Var(0, inf, False, 'w')
    solver.Minimize(w)
    # solver.Minimize(sum(f_cost[i]*ys[i] for i in range(facilities)) + sum(ws))
    status = solver.Solve()
    assert status == solver.OPTIMAL

    slave = pywraplp.Solver('CFL worker', GLOP_SOLVER)
    print("1")
    # slave.EnableOutput()
    # qs = [slave.Var(ys[j].solution_value(), ys[j].solution_value(), False, f'q[{j}]') for j in range(facilities)]
    qs = [slave.Var(-inf, inf, False, f'q[{j}]') for j in range(facilities)]
    print("2")
    q_geq = [slave.Constraint() for j in range(facilities)]
    print("3")
    for j in range(facilities):
        q_geq[j].SetCoefficient(qs[j], 1)
    print("4")
    # q_leq = [slave.Constraint() for j in range(facilities)]

    xs = [[slave.Var(0, 1, False, f'x{i}{j}') for j in range(facilities)] for i in range(customers)]
    print("5")
    for i in range(customers):
        slave.Add(sum(xs[i]) == 1)
        for j in range(facilities):
            slave.Add(xs[i][j] - qs[j] <= 0)
    print("6")
    for j in range(facilities):
        slave.Add(sum(demand[i] * xs[i][j] for i in range(customers)) - capacities[j] * qs[j] <= 0)

    print("7")
    slave.Minimize(sum(f_cost[j] * qs[j] for j in range(facilities)) +
                   sum(sum(demand[i] * (s_cost + j) * xs[i][j] for j in range(facilities)) for i in range(customers)))

    print("8")
    prev_obj = -float('inf')
    terminate = False
    while not terminate:
        # kp = []
        # for j in range(customers):
        #     pass
        # solver.Add(w >= 0)
        for j in range(facilities):
            # q_geq[j].SetBounds(ys[j].solution_value(), inf)
            q_geq[j].SetBounds(ys[j].solution_value(), ys[j].solution_value())
            # q_leq[j].SetBounds(-inf, ys[j].solution_value())

        print("9")

        # Reduced cost: negative coeffs in obj <==> basis matrix OR dual problem

        status = slave.Solve()
        if status != solver.OPTIMAL:
            print()
        print("10")
        # print_solution(xs, ys)

        if prev_obj == slave.Objective().Value():
            terminate = True
        else:
            prev_obj = slave.Objective().Value()
            solver.Add(w >= slave.Objective().Value() + sum(q_geq[j].DualValue() * (ys[j] - ys[j].solution_value()) for j in range(facilities)))
        print("11")
        solver.Solve()
        print("12")
        assert status == solver.OPTIMAL

    # return solver.Objective().Value()
    for y in ys:
        y.SetInteger(True)
    solver.Solve()
    prev_obj = -float('inf')
    terminate = False
    while not terminate:
        # kp = []
        # for j in range(customers):
        #     pass
        # solver.Add(w >= 0)
        for j in range(facilities):
            # q_geq[j].SetBounds(ys[j].solution_value(), inf)
            q_geq[j].SetBounds(ys[j].solution_value(), ys[j].solution_value())
            # q_leq[j].SetBounds(-inf, ys[j].solution_value())

        # Reduced cost: negative coeffs in obj <==> basis matrix OR dual problem

        slave.Solve()
        # print_solution(xs, ys)

        if prev_obj == slave.Objective().Value():
            terminate = True
        else:
            prev_obj = slave.Objective().Value()
            solver.Add(w >= slave.Objective().Value() + sum(q_geq[j].DualValue() * (ys[j] - ys[j].solution_value()) for j in range(facilities)))
        status = solver.Solve()

    assert status == pywraplp.Solver.OPTIMAL

    print(solver.Objective().Value())
    return solver.Objective().Value()

if __name__ == '__main__':
    # start = time()
    # v1 = base()
    # end = time()
    # print(end - start)
    v2 = main()
    end2 = time()
    print(end2 - end)
    # assert v1 - v2 <= 0.0001
