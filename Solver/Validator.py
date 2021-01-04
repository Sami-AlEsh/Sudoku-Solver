def isValidSudoku(board):
    """
    :type board: List[List[int]]
    :rtype: bool
    """
    for i in range(9):
        row = {}
        column = {}
        block = {}
        row_cube = 3 * (i // 3)
        column_cube = 3 * (i % 3)
        for j in range(9):
            if board[i][j] != 0 and board[i][j] in row:
                return False
            row[board[i][j]] = 1
            if board[j][i] != 0 and board[j][i] in column:
                return False
            column[board[j][i]] = 1
            rc = row_cube + j // 3
            cc = column_cube + j % 3
            if board[rc][cc] in block and board[rc][cc] != 0:
                return False
            block[board[rc][cc]] = 1
    return True


# print(isValidSudoku([
#    [8, 2, 7, 1, 5, 4, 3, 9, 6],
#    [9, 6, 5, 3, 2, 7, 1, 4, 8],
#    [3, 4, 1, 6, 8, 9, 7, 5, 2],
#    [5, 9, 3, 4, 6, 8, 2, 7, 1],
#    [4, 7, 2, 5, 1, 3, 6, 8, 9],
#    [6, 1, 8, 9, 7, 2, 4, 3, 5],
#    [7, 8, 6, 2, 3, 5, 9, 1, 4],
#    [1, 5, 4, 7, 9, 6, 8, 2, 3],
#    [2, 3, 9, 8, 4, 1, 5, 6, 7]]))
