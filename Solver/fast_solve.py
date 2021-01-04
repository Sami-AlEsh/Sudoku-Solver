# written by Nathan Esau, Aug 2020
import copy

def get_possible_values(board, row, col):
    possible_values = [1,2,3,4,5,6,7,8,9]
    tl_row = row - row % 3
    tl_col = col - col % 3
    for i in range(9):
        if len(board[row][i]) == 1: # entry already solved
            if board[row][i][0] in possible_values:
                possible_values.remove(board[row][i][0])
        if len(board[i][col]) == 1: # entry already solved
            if board[i][col][0] in possible_values:
                possible_values.remove(board[i][col][0])
        if len(board[tl_row+i//3][tl_col+i%3]) == 1: # entry already solved
            if board[tl_row+i//3][tl_col+i%3][0] in possible_values:
                possible_values.remove(board[tl_row+i//3][tl_col+i%3][0])
    return possible_values

def fill_missing_entries(board, box):
    row_start, row_end = (0,2) if box in [0,1,2] else (3,5) if box in [3,4,5] else (6,8)
    col_start, col_end = (0,2) if box in [0,3,6] else (3,5) if box in [1,4,7] else (6,8)
    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            if len(board[row][col]) == 1: # entry already solved
                continue
            board[row][col] = get_possible_values(board, row, col)

def solve_missing_entries(board, box):
    row_start, row_end = (0,2) if box in [0,1,2] else (3,5) if box in [3,4,5] else (6,8)
    col_start, col_end = (0,2) if box in [0,3,6] else (3,5) if box in [1,4,7] else (6,8)
    possible_squares = dict((i, []) for i in range(1, 10, 1))
    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            for e in board[row][col]:
                possible_squares[e].append((row, col))
    for (k, v) in possible_squares.items():
        if len(v) == 1:
            row, col = v[0]
            if len(board[row][col]) != 1: # solve entry
                board[row][col] = [k]

def solve_strategy(board):
    for _ in range(25): # max_iter = 25
        initial_board = copy.deepcopy(board)
        for box in range(9):
            fill_missing_entries(board, box)
            solve_missing_entries(board, box)
        if board == initial_board:
            return "stuck"
        solved = True
        for i in range(9):
            for j in range(9):
                if len(board[i][j]) == 0:
                    return "failed"
                if len(board[i][j]) != 1:
                    solved = False
        if solved:
            return "solved"

def get_guess(board):
    solved_count = {}
    for i in range(9): # row i, col i, box i
        rc, cc, bc = 0, 0, 0
        for j in range(9):
            if len(board[i][j]) == 1:
                rc += 1
            if len(board[j][i]) == 1:
                cc += 1
            if len(board[i//3*3 + j//3][i%3*3 + j%3]) == 1:
                bc += 1
        if rc < 9: solved_count["r"+str(i)] = rc
        if cc < 9: solved_count["c"+str(i)] = cc
        if bc < 9: solved_count["b"+str(i)] = bc
    rcb = max(solved_count, key=solved_count.get)
    square = None
    options = None
    t, i = rcb[0], int(rcb[1])
    for j in range(9):
        if t == 'r' and len(board[i][j]) > 1:
            square, options = [i,j], board[i][j]
            break
        if t == 'c' and len(board[j][i]) > 1:
            square, options = [j,i], board[j][i]
            break
        if t == 'b' and len(board[i//3*3+j//3][i%3*3+j%3]) > 1:
            square, options = [i//3*3+j//3, i%3*3+j%3], board[i//3*3+j//3][i%3*3+j%3]
            break
    return {"rcb": rcb, "square": square, "options": options}

def apply_guess(board, guess, value):
    square = guess["square"]
    board[square[0]][square[1]] = [value]

def solve(initial_board): # return solved board
    board = copy.deepcopy(initial_board)
    root = {"board":board,"parent":None,"child":None,"depth":0,"guess":None,"value":None}
    node = root
    while True:
        state = solve_strategy(board)
        if state == "solved":
            return board
        if state == "stuck":                
            node["board"] = copy.deepcopy(board)
            node["child"] = {"board": board, "parent": node, "depth": root["depth"] + 1}
            node = node["child"]
            node["guess"] = get_guess(board)
            node["value"] = node["guess"]["options"][0]
            apply_guess(board, node["guess"], node["value"])
        if state == "failed": # backtrack - change guess
            while len(node["guess"]["options"]) <= 1:
                node = node["parent"]
            board = copy.deepcopy(node["parent"]["board"])
            node["board"] = copy.deepcopy(board)
            node["guess"]["options"] = node["guess"]["options"][1:]
            node["value"] = node["guess"]["options"][0]
            apply_guess(board, node["guess"], node["value"])

def print_board(board):
    for i in range(9):
        for j in range(9):
            if len(board[i][j]) == 1:
                print(board[i][j][0], end= " ")
            else:
                print("X", end=" ")
        print("")
