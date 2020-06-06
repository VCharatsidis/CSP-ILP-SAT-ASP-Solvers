###
### Propagation function to be used in the recursive sudoku solver
###
def propagate(sudoku_possible_values,k):
    # print(sudoku_possible_values)
    # print(len(sudoku_possible_values))
    # print(len(sudoku_possible_values[0]))
    # print(len(sudoku_possible_values[0][0]))

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

    for i1 in range(k):
        for j1 in range(k):
            certain_values = [];
            for i2 in range(k):
                for j2 in range(k):
                    i = i1 * k + i2;
                    j = j1 * k + j2;
                    possibilities = sudoku_possible_values[i][j];
                    if len(possibilities) == 1:
                        value = possibilities[0];
                        if value in certain_values:
                            return True;
                        else:
                            certain_values.append(value);


    # Remove all the numbers that cannot be used from the possibilities because exist in row and col
    for i in range(k ** 2):
        for j in range(k ** 2):
            to_be_removed = []
            if(len(sudoku_possible_values[i][j])) > 1:
                for possibility in sudoku_possible_values[i][j]:
                    if possibility in used_row[i]:
                        to_be_removed.append(possibility)

            sudoku_possible_values[i][j] = [x for x in sudoku_possible_values[i][j] if x not in to_be_removed]

            to_be_removed = []
            if (len(sudoku_possible_values[j][i])) > 1:
                for possibility in sudoku_possible_values[j][i]:
                    if possibility in used_col[i]:
                        to_be_removed.append(possibility)

            sudoku_possible_values[j][i] = [x for x in sudoku_possible_values[j][i] if x not in to_be_removed]


    # print(sudoku_possible_values)
    # print(len(sudoku_possible_values))
    # print(len(sudoku_possible_values[0]))
    # print(len(sudoku_possible_values[0][0]))
    # input()
    return sudoku_possible_values;

###
### Solver that uses SAT encoding
###
def solve_sudoku_SAT(sudoku,k):
    return None;

###
### Solver that uses CSP encoding
###
def solve_sudoku_CSP(sudoku,k):
    return None;

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