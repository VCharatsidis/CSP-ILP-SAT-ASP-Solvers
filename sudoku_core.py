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

        not_assigned = int(flatten_sudoku[i - 1]) == 0

        if not_assigned:
            for c in range(1, number_colors + 1):
                clause.append(var_number(i, c))
        else:
            clause.append(var_number(i, int(flatten_sudoku[i - 1])))

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
    # First we create a list containing all the edges for all vertices in the sudoku
    length = len(sudoku)
    num_vertices = length ** 2
    matrix = np.arange(num_vertices).reshape(length, length)
    # edges = {'squares':[], 'rows':[], 'columns':[]}
    edges = []
    sudoku = np.array(sudoku).reshape(length * length)

    # The loop below fills the edges list with all edges in the sudoku
    for i in range(length):
        for j in range(length):

            # specify the current value i,j as the left-hand value for the edge tuple
            left = int(matrix[i][j] + 1)

            # iterate over values in the square
            col = j // k
            row = i // k
            rows = matrix[(row * k):(row * k + k)]
            box = [x[(col * k):(col * k + k)] for x in rows]
            for v in range(k):
                for w in range(k):
                    right = int(box[v][w] + 1)

                    # make sure that we're not assigning the current value as the right-hand vertix
                    if (row * k + v, col * k + w) != (i, j):
                        if (right, left) not in edges:
                            edges.append((left, right))

            # iterative over cells in row i,
            for g in range(length):
                right = int(matrix[i][g] + 1)
                if (i, g) != (i, j):
                    if (right, left) not in edges:
                        edges.append((left, right))

            # iterate over cells in column j,
            for c in range(length):
                right = int(matrix[c][j] + 1)
                if (c, j) != (i, j):
                    if (right, left) not in edges:
                        # edges['columns'].append((left, right))
                        edges.append((left, right))

    # for each variable in the sudoku we set a domain d {1,2,.....,9}, except for the variables
    # for which we already have a fixed value

    print(len(edges))
    # sys.exit()
    # max_val = sum(range(0,length+1))
    # print(max_val)
    model = cp_model.CpModel()
    vars = dict()

    # domain = {}
    # for i in range(1, num_vertices +1):
    #     domain[i] = []
    #
    # boxes = []
    # for i in range(k):
    #     for j in range(k):
    #         box = []
    #         rows = matrix[i*k:(i+1)*k]
    #         for row in rows:
    #             subrow = row[j*k:(j+1)*k].tolist()
    #             for x in subrow:
    #                 box.append(sudoku[x])
    #         boxes.append(box)
    #
    # rows = []
    # for i in range(length):
    #     row = []
    #     for j in range(length):
    #         row.append(sudoku[matrix[i][j]])
    #     # for idx in range(1, idxs)
    #     rows.append(row)
    #
    # columns = []
    # for i in range(length):
    #     column = []
    #     for j in range(length):
    #         column.append(sudoku[matrix[j][i]])
    #     # for idx in range(1, idxs)
    #     columns.append(column)
    #
    # columns = columns * length
    #
    # for i in range(num_vertices):
    #     row_idx = i // length
    #     for x in rows[row_idx]:
    #         domain[i+1].append(x)
    #
    # for i in range(num_vertices):
    #     for x in columns[i]:
    #         domain[i+1].append(x)
    #
    # # for i in range(1, length+1):
    # #     for j in range(1, length+1):
    # #         idx = i*j
    # #         for x in boxes[i-1]:
    # #             domain[idx].append(x)
    #
    # print(set(domain[2]))
    #
    #
    # # for i in range(1, length+1):
    # #     for j in range(1, length+1):
    # #         idx = i*j
    # #         for x in boxes[i-1]:
    # #             domain[idx].append(x)
    #
    #
    #         # for x in rows[i-1]:
    #         #     domain[idx].append(x)
    #         # for x in columns[i-1]:
    #         #     domain[idx].append(x)
    #
    #         # domain[idx].append(rows[i-1])
    #
    #         # domain[idx].append(columns)
    #
    #     # print(set(domain[24]))
    #     # sys.exit()
    #
    #
    #
    # Set domains for each variable
    for i in range(1, num_vertices + 1):
        sudoku_val = int(sudoku[i - 1])
        if sudoku_val == 0:
            vars[i] = model.NewIntVar(1, length, "x{}".format(i))
        else:
            vars[i] = model.NewIntVar(sudoku_val, sudoku_val, "x{}".format(i))
            # vars[i] = model.NewIntVar(1, 9, "x{}".format(i))

    for (i, j) in edges:
        # model.AddAllDifferent([vars[i], vars[j]])
        model.Add(vars[i] != vars[j])
    # for i in range(length):
    #     row = []
    #     for j in range(length):
    #         row.append(vars[matrix[i][j]+1])
    #     model.Add(sum(row) == max_val)
    #     model.AddAllDifferent(row)
    #
    # for i in range(length):
    #     col = []
    #     for j in range(length):
    #         col.append(vars[matrix[j][i]+1])
    #     model.Add(sum(col) == max_val)
    #     model.AddAllDifferent(col)
    #
    # for i in range(k):
    #     for j in range(k):
    #         rows = matrix[i*k:(i+1)*k]
    #         box = []
    #         for row in rows:
    #             subrow = row[j*k:(j+1)*k].tolist()
    #             for x in subrow:
    #                 box.append(vars[x+1])
    #         model.Add(sum(box) == max_val)
    #         model.AddAllDifferent(box)
    # print('start solving')

    solver = cp_model.CpSolver()
    answer = solver.Solve(model)
    matrix = matrix.reshape(length ** 2)
    if answer == cp_model.FEASIBLE:
        for i in range(1, num_vertices + 1):
            matrix[i - 1] = solver.Value(vars[i])
        return matrix.reshape(length, length).tolist()
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