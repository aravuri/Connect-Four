import numpy as np
from time import sleep
import pygame
import matplotlib.pyplot as plt
import torch

class Board:
    def __init__(self):
        self.board = np.zeros((7, 6), dtype=int)

    def put(self, piece, column):
        if not np.any(self.board[column] == 0):
            return False
        row = (self.board[column] == 0).argmax(axis=0)
        self.board[column][row] = piece
        return True

    def getWinner(self):
        maskP1 = self.board == 1
        maskP2 = self.board == 2
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                subMask1 = maskP1[i:i+4, j:j+4]
                subMask2 = maskP2[i:i+4, j:j+4]
                if max([np.sum(subMask1[0]), np.sum(subMask1[:, 0]),
                        np.trace(subMask1), np.trace(np.fliplr(subMask1))]) == 4:
                    return 1
                if max([np.sum(subMask2[0]), np.sum(subMask2[:, 0]),
                        np.trace(subMask2), np.trace(np.fliplr(subMask2))]) == 4:
                    return 2
        return 0

    def drawBoard(self, screen):
        dispBoard = np.flipud(self.board.T)
        WHITE = pygame.Color(255, 255, 255)
        RED = pygame.Color(255, 0, 0)
        YELLOW = pygame.Color(255, 255, 0)
        BLUE = pygame.Color(0, 0, 255)
        screen.fill(BLUE)
        for i in range(dispBoard.shape[1]):
            for j in range(dispBoard.shape[0]):
                color = WHITE if dispBoard[j][i] == 0 else RED if dispBoard[j][i] == 1 else YELLOW
                pygame.draw.circle(screen, color, (i*100 + 50, j*100 + 50), 50)

    def play(self, agentP1, agentP2):
        pygame.init()
        screen = pygame.display.set_mode((700, 600))
        p1turn = True
        pygame.display.set_caption("Player " + str(1 if p1turn else 2) + " Turn")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            self.drawBoard(screen)
            pygame.display.update()
            (agentP1 if p1turn else agentP2).make_move(self, 1 if p1turn else 2)
            p1turn = not p1turn
            winner = self.getWinner()
            if winner != 0:
                sleep(0.2)
                print("Player " + str(winner) + " Wins!")
                self.drawBoard(screen)
                pygame.display.update()
                sleep(10)
                quit()
            sleep(0.2)

    def getBoard(self):
        x = torch.from_numpy(self.board)
        x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 0)
        x = x.type(torch.FloatTensor)
        return x

