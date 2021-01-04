# to see print output, use "py.test -s"
import pytest
import slow_solve

# takes > 1s to solve this board
# by comparison fast solve can solve this in < 0.1s
def test_solve():
    board = [[0,1,9,0,0,2,0,0,0],
             [0,0,0,0,0,8,1,9,0],
             [3,0,0,0,0,0,0,2,0],
             [2,7,0,0,0,0,5,0,0],
             [0,0,0,0,4,9,0,0,0],
             [5,0,0,0,0,0,0,1,3],
             [0,0,0,0,0,0,0,4,6],
             [0,0,3,0,0,1,9,8,0],
             [0,0,0,0,7,6,0,0,0]]
    slow_solve.solve(board)
    print(board)
