from graphics import Board
import numpy as np

class Agent():
    def __init__(self):
        pass

    def make_move(self, board, player):
        pass


class RandomAgent(Agent):
    def make_move(self, board, player):
        while not board.put(player, np.random.randint(7)):
            pass


class HumanAgent(Agent):
    def make_move(self, board, player):
        column = int(input("What Column? "))
        board.put(player, column)


class BotAgent(Agent):
    def __init__(self, policy_net):
        super().__init__()
        self.policy_net = policy_net

    def make_move(self, board, player):
        column = self.select_best_action(board.getBoard(), self.policy_net)
        board.put(player, column)

    def select_best_action(self, state, policy_net):
        return policy_net(state).max(1)[1].view(1, 1)