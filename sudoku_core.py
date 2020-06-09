from pysat.formula import CNF
from pysat.solvers import MinisatGH
import numpy as np
# CSP dependencies
from ortools.sat.python import cp_model
import clingo
import gurobipy as gp
from gurobipy import GRB



def make_sudoku_graph(k):
    vertices = []
    for row in range(k ** 2):
        row = list(range((row * k ** 2), (row + 1) * (k ** 2)))
        vertices.append(row)

    edges = []
    for i in range(k ** 2):
        for j in range(k ** 2):

            vertex_1 = vertices[i][j] + 1

            # connect vertices in the same row
            for col in range(k ** 2):
                vertex_2 = vertices[i][col] + 1
                if vertex_2 != vertex_1:
                    edge = (vertex_1, vertex_2)
                    rev_edge = (vertex_2, vertex_1)
                    if edge not in edges and rev_edge not in edges:
                        edges.append(edge)

            # connect vertices in the same col
            for row in range(k ** 2):
                vertex_2 = vertices[row][j] + 1
                if vertex_2 != vertex_1:
                    edge = (vertex_1, vertex_2)
                    rev_edge = (vertex_2, vertex_1)
                    if edge not in edges and rev_edge not in edges:
                        edges.append(edge)

            # connect vertices in the same box
            col = j // k
            row = i // k
            for bi in range(k):
                for bj in range(k):
                    vertex_2 = vertices[row * k + bi][col * k + bj] + 1
                    if vertex_2 != vertex_1:
                        edge = (vertex_1, vertex_2)
                        rev_edge = (vertex_2, vertex_1)
                        if edge not in edges and rev_edge not in edges:
                            edges.append(edge)

    return vertices, edges


###
### Propagation function to be used in the recursive sudoku solver
###
def propagate(sudoku_possible_values,k):
    # Find all the numbers that cannot be used per row
    used_row = []
    for i in range(k**2):
        row = []
        for j in range(k**2):
            possibilities = sudoku_possible_values[i][j];
            if len(possibilities) == 1:
                row.append(possibilities[0])

        used_row.append(row)

    # Find all the numbers that cannot be used per col
    used_col = []
    for i in range(k ** 2):
        col = []
        for j in range(k ** 2):
            possibilities = sudoku_possible_values[j][i];
            if len(possibilities) == 1:
                col.append(possibilities[0])

        used_col.append(col)

    # Find all the numbers that cannot be used per square
    used_square = []
    for i1 in range(k):
        for j1 in range(k):
            square = []
            for i2 in range(k):
                for j2 in range(k):
                    i = i1 * k + i2
                    j = j1 * k + j2
                    possibilities = sudoku_possible_values[i][j]
                    if len(possibilities) == 1:
                        square.append(possibilities[0])

            used_square.append(square)

    # Remove all the numbers that cannot be used from the possibilities because exist in row, col and box.
    # Adding a number in the prohibited if a cell has left one value possible.
    # Repeat this refinement until we cannot remove any more values from the possibilities.
    change = True
    while change:
        change = False
        for i in range(k ** 2):
            for j in range(k ** 2):
                to_be_removed = []
                # Remove numbers that cannot be used because:
                if(len(sudoku_possible_values[i][j])) > 1:
                    for possibility in sudoku_possible_values[i][j]:
                        # they exist in the same row
                        if possibility in used_row[i]:
                            to_be_removed.append(possibility)

                        # they exist in the same col
                        if possibility in used_col[j]:
                            to_be_removed.append(possibility)

                        # they exist in the same box
                        if possibility in used_square[(i // k) * k + j // k]:
                            to_be_removed.append(possibility)

                    sudoku_possible_values[i][j] = [x for x in sudoku_possible_values[i][j] if x not in to_be_removed]

                    # if after the deletion of impossible values the cell is left with 1 possibility
                    # append it in the list of unusables for other cells
                    # and stay in the while loop for another refinement.
                    if(len(sudoku_possible_values[i][j])) == 1:
                        change = True
                        certain_value = sudoku_possible_values[i][j][0]
                        used_row[i].append(certain_value)
                        used_col[j].append(certain_value)
                        used_square[(i // k) * k + j // k].append(certain_value)

    return sudoku_possible_values;




###
### Solver that uses SAT encoding
###
def solve_sudoku_SAT(sudoku, k):
    num_vertices = k ** 4
    vertices, edges = make_sudoku_graph(k)

    number_colors = k ** 2
    formula = CNF()

    # Assign a positive integer for each propositional variable.
    def var_number(i, c):
        return ((i - 1) * number_colors) + c

    flatten_sudoku = np.array(sudoku).reshape(num_vertices)

    # Clause that ensures that each vertex there is one value.
    for i in range(1, num_vertices + 1):
        clause = []

        sudoku_value = int(flatten_sudoku[i - 1])
        not_assigned = sudoku_value == 0

        if not_assigned:
            for c in range(1, number_colors + 1):
                clause.append(var_number(i, c))
        else:
            clause.append(var_number(i, sudoku_value))

        formula.append(clause)

    # Ensure that only one value is assigned.
    for i in range(1, num_vertices + 1):
        for c1 in range(1, number_colors + 1):
            for c2 in range(c1 + 1, number_colors + 1):
                clause = [-1 * var_number(i, c1), -1 * var_number(i, c2)]
                formula.append(clause)

    # Ensure that the rules of sudoku are kept, no adjacent vertices should have the same color/value.
    for (v1, v2) in edges:
        for c in range(1, number_colors + 1):
            clause = [-1 * var_number(v1, c), -1 * var_number(v2, c)]
            formula.append(clause)

    solver = MinisatGH()
    solver.append_formula(formula)
    answer = solver.solve()

    flatten_vertices = np.array(vertices).reshape(num_vertices)

    if answer:
        print("The sudoku is solved.")
        model = solver.get_model()
        print(model)
        for i in range(1, num_vertices + 1):
            for c in range(1, number_colors + 1):
                if var_number(i, c) in model:
                    flatten_vertices[i - 1] = c

        return flatten_vertices.reshape(k**2, k**2).tolist()
    else:
        print("The sudoku has no solution.")
        return None







###
### Solver that uses CSP encoding
###
def solve_sudoku_CSP(sudoku,k):
    num_vertices = k ** 4
    vertices, edges = make_sudoku_graph(k)
    num_colors = k**2
    model = cp_model.CpModel()
    vars = dict()
    flatten_sudoku = np.array(sudoku).reshape(num_vertices)

    for i in range(1, num_vertices+1):
        sudoku_value = int(flatten_sudoku[i - 1])
        not_assigned = sudoku_value == 0

        if not_assigned:
            vars[i] = model.NewIntVar(1, num_colors, "x{}".format(i))
        else:
            vars[i] = model.NewIntVar(sudoku_value, sudoku_value, "x{}".format(i))

    for (i, j) in edges:
        model.Add(vars[i] != vars[j])

    solver = cp_model.CpSolver()
    answer = solver.Solve(model)

    flatten_vertices = np.array(vertices).reshape(num_vertices)

    if answer == cp_model.FEASIBLE:
        for i in range(1, num_vertices + 1):
            flatten_vertices[i - 1] = solver.Value(vars[i])
        return flatten_vertices.reshape(k**2, k**2).tolist()
    else:
        return None







###
### Solver that uses ASP encoding
###
def solve_sudoku_ASP(sudoku,k):
    num_vertices = k ** 4
    vertices, edges = make_sudoku_graph(k)

    number_colors = k ** 2

    asp_code = """"""
    for row in vertices:
        for cell in row:
            asp_code += "vertex(v" + str(cell+1) + ").\n"

    for edge in edges:
        asp_code += "edge(v" + str(edge[0]) + ",v"+str(edge[1]) + ").\n"

    # # Add constraints for assigned values
    # flatten_sudoku = np.array(sudoku).reshape(num_vertices)
    #
    # counter = 0
    # for row in vertices:
    #     for cell in row:
    #         if flatten_sudoku[counter] != 0:
    #             asp_code += "color(v" + str(cell+1) + "," + str(flatten_sudoku[counter]) + ").\n"
    #         counter += 1

    for c in range(number_colors):
        asp_code += "color(V," + str(c+1) + ") :- vertex(V)"
        for c_other in range(number_colors):
            if c != c_other:
                asp_code += ", not color(V," + str(c_other+1) + ")"

        asp_code += ".\n"

    # Add constraints for assigned values
    flatten_sudoku = np.array(sudoku).reshape(num_vertices)

    counter = 0
    for row in vertices:
        for cell in row:
            if flatten_sudoku[counter] != 0:
                asp_code += "color(v" + str(cell + 1) + "," + str(flatten_sudoku[counter]) + ").\n"
            counter += 1

    asp_code += """:- edge(V1,V2), color(V1,C), color(V2,C).\n"""

    asp_code += """#show color/2."""

    # print(asp_code)
    # print(flatten_sudoku)

    control = clingo.Control()
    control.add("base", [], asp_code)
    control.ground([("base", [])])

    def on_model(model):
        print(model.symbols(shown=True))

    control.configuration.solve.models = 0
    answer = control.solve(on_model=on_model)


    if answer.satisfiable == True:
        print("sudoku solution")
    else:
        print("There is not solution")

    print(asp_code)
    input()
    return None;








###
### Solver that uses ILP encoding
###
def solve_sudoku_ILP(sudoku,k):
    model = gp.Model()
    vertices, edges = make_sudoku_graph(k)
    num_vertices = k**4
    num_colors = k**2

    flatten_sudoku = np.array(sudoku).reshape(num_vertices)

    # Make variables
    vars = dict()
    for i in range(1, num_vertices + 1):
        sudoku_value = int(flatten_sudoku[i - 1])
        not_assigned = sudoku_value == 0

        if not_assigned:
            for c in range(1, num_colors + 1):
                vars[(i, c)] = model.addVar(vtype=GRB.BINARY, name="x({},{})".format(i, c))
        else:
            vars[(i, sudoku_value)] = model.addVar(vtype=GRB.BINARY, name="x({},{})".format(i, sudoku_value))

    # Add constraints
    for i in range(1, num_vertices + 1):
        sudoku_value = int(flatten_sudoku[i - 1])
        not_assigned = sudoku_value == 0

        if not_assigned:
            model.addConstr(gp.quicksum([vars[(i, c)] for c in range(1, num_colors + 1)]) == 1)
        else:
            model.addConstr(gp.quicksum([vars[(i, sudoku_value)]]) == 1)

    # Ensure that the values are proper.
    for (v1, v2) in edges:
        for c in range(1, num_colors + 1):
            if (v1, c) in vars.keys() and (v2, c) in vars.keys():
                model.addConstr(vars[(v1, c)] + vars[(v2, c)] <= 1)

    model.optimize()

    flatten_vertices = np.array(vertices).reshape(num_vertices)

    if model.status == GRB.OPTIMAL:
        for i in range(1, num_vertices + 1):
            for c in range(1, num_colors + 1):
                if (i, c) in vars.keys():
                    if vars[(i, c)].x == 1:
                        flatten_vertices[i - 1] = c
        return flatten_vertices.reshape(k**2, k**2).tolist()

    else:
        return None
