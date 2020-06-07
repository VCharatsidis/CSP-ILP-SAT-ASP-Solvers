from pysat.formula import CNF
from pysat.solvers import MinisatGH
import numpy as np
# CSP dependencies
from ortools.sat.python import cp_model

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
                # Remove numbers that cannot be used because they exist in the same row
                if(len(sudoku_possible_values[i][j])) > 1:
                    for possibility in sudoku_possible_values[i][j]:
                        if possibility in used_row[i]:
                            to_be_removed.append(possibility)

                        if possibility in used_col[j]:
                            to_be_removed.append(possibility)

                        if possibility in used_square[(i // k) * k + j // k]:
                            to_be_removed.append(possibility)

                    sudoku_possible_values[i][j] = [x for x in sudoku_possible_values[i][j] if x not in to_be_removed]

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
    vertices = []
    for row in range(k**2):
        row = list(range((row * k**2), (row+1)*(k**2)))
        vertices.append(row)

    edges = []
    for i in range(k ** 2):
        for j in range(k ** 2):

            vertex_1 = vertices[i][j]+1

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
    for (i1, i2) in edges:
        for c in range(1, number_colors + 1):
            clause = [-1 * var_number(i1, c), -1 * var_number(i2, c)]
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

    solver = cp_model.CpSolver();
    answer = solver.Solve(model);

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
    return None;

###
### Solver that uses ILP encoding
###
def solve_sudoku_ILP(sudoku,k):
    return None;