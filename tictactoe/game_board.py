# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:49:57 2017

@author: Edward
"""

import copy
import numpy as np
import matplotlib.pyplot as plt

class GameBoard():
    def __init__(self, start_player=1):
        # creates the board
        # players are defined as 1 and -1
        self.board = np.zeros((3, 3))
        self.current_player = start_player
        self.history = []
        
    def has_won(self, player):
        # checks if a player has won
        # check rows
        row_sums = np.sum(self.board*player, axis=1)
        if np.max(row_sums) == 3: 
            print('rows')
            return True     
        # check columns
        col_sums = np.sum(self.board*player, axis=0)
        if np.max(col_sums) == 3: 
            print('cols')
            return True
        # check diagonal
        if np.trace(self.board*player) == 3: 
            print('trace')
            return True
        
        if np.trace(self.board[::-1,:]*player) == 3:
            print('trace2')
            return True
        
        return False
    
    def has_lost(self, player):
        return self.has_won(-1*player)
    
    def game_over(self):        
        return not 0.0 in self.board or self.has_won(1) or self.has_won(-1)
    
    def reset_board(self, next_starting_player=1):
        self.board = np.zeros((3, 3))
        self.current_player = next_starting_player
        self.history = []
        
    def valid_move(self, i, j):
        return self.board[i,j]==0.0
    
    def make_move(self, i, j):       
        self.history.append(((copy.copy(self.board), self.current_player), (i, j)))
        self.board[i, j] = self.current_player
        self.current_player *= -1
        
    def get_pieces(self):
        return [p for p in self.board.flatten()]
        
    def plotboard(self):
        plt.figure()
        for i,((b,_),(_,_)) in enumerate(self.history,1):
            plt.subplot(2,5,i)
            plt.imshow(b)
    
        plt.subplot(2,5,i+1)
        plt.imshow(self.board)
        
        