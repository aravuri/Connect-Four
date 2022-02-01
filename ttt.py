import numpy as np

def put(board, location, color):
    board[location / 3][location % 3] = color

def winner(board):
    if (np.sum(board,axis=0)==3).any():
        return 1
    elif (np.sum(board,axis=0)==-3).any():
        return -1
    elif (np.sum(board,axis=1)==3).any():
        return 1
    elif (np.sum(board,axis=1)==-3).any():
        return -1
    elif np.trace(board) == 3:
        return 1
    elif np.trace(board) == -3:
        return -1
    elif np.trace(np.fliplr(board)) == 3:
        return 1
    elif np.trace(np.fliplr(board)) == -3:
        return -1
    return 0


def dfs(board, count, turn):
    if (w := winner(board)) != 0:
        return w
    if count > 9:
        return 0
    for i in range(9):
        put(board, i, turn)
        dfs(board, count + 1, -1 if turn == 1 else 1)
        put(board, i, 0)


