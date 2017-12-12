# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:30:59 2017

@author: Edward
"""
import sys
import copy

import numpy as np
import matplotlib.pyplot as plt
import random

from PyQt5.QtWidgets import (QWidget, QPushButton, QFrame, QApplication, QDialog, QMessageBox)
from PyQt5.QtGui import QColor, QPixmap, QIcon
from PyQt5.QtCore import QSize

from game_board import GameBoard
from policy import MovementPolicy
        
import itertools

class GameUI(QWidget):
    def __init__(self):
        super().__init__()
        
        self.pieces = []
        self.col = QColor(0, 0, 0)  
        locs = [10, 120, 230]
        self.o_pix = QIcon('O.png')
        self.x_pix = QIcon('X.png')
        self.b_pix = QIcon('B.png')
        
        for i, (y, x)  in enumerate(itertools.product(locs, locs)):
            game_piece = QPushButton(self)
            game_piece.setGeometry(x, y, 100, 100)
            game_piece.setIcon(self.b_pix)
            game_piece.setIconSize(QSize(100,100))
            game_piece.id = i
            game_piece.clicked.connect(self.piece_pressed)
            self.pieces.append(game_piece)
            
            
        self.setGeometry(300, 300, 340, 380)
        self.setFixedSize(self.size())
        self.setWindowTitle('3T-RL')
        self.show()

        # Initialize the gameboard
        self.pos_lookup = {i:(x,y) for i,(x,y) in enumerate(itertools.product(range(3), range(3)))}        
        self.board = GameBoard()
        self.policy = MovementPolicy()

    def piece_pressed(self):
        print('Piece Pressed!', self.sender().id, self.pos_lookup[self.sender().id])
        
        move_i, move_j = self.pos_lookup[self.sender().id]
        if self.board.valid_move(move_i, move_j):
            self.board.make_move(move_i, move_j)
            self.update_board_view()
        else:
            return

        
        if self.board.game_over():
            self.game_over_popup()
            return 
        
        move_i, move_j = self.policy.get_next_move(self.board.board, self.board.current_player)
        if self.board.valid_move(move_i, move_j):
            self.board.make_move(move_i, move_j)
            self.update_board_view()
        else:
            print('Invalid move made by AI, this should not happen!')
            assert(0)
            
        if self.board.game_over():
            self.game_over_popup()
            return 
            
    def game_over_popup(self):
        result = 0
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle('Game Over')
        msg.setStandardButtons(QMessageBox.Ok)
        if self.board.has_won(1):
            msg.setText('X has won!')
            result = 1
        elif self.board.has_won(-1):
            result = -1
            msg.setText('O has won!')
        else:
            msg.setText('The game ended in a draw')
        msg.exec_()
        self.reset_board(result)
    
    def update_board_view(self):
        piece_map = {0: self.b_pix, -1: self.o_pix,1: self.x_pix}
        
        for i, piece_type in enumerate(self.board.get_pieces()):
            self.pieces[i].setIcon(piece_map[int(piece_type)])
                                
    def reset_board(self, result):
        for piece in self.pieces:
            piece.setIcon(self.b_pix)
        
        self.policy.update(self.board.history, result)
        
        next_player = random.choice([1,-1])
        print('Next player is', next_player)
        self.board.reset_board(next_starting_player=next_player)
        
        if next_player == -1:
            move_i, move_j = self.policy.get_next_move(self.board.board, self.board.current_player)
            if self.board.valid_move(move_i, move_j):
                self.board.make_move(move_i, move_j)
                self.update_board_view()
            else:
                print('Invalid move made by AI, this should not happen!')
                assert(0)
        
if __name__ == '__main__':

    
    # GUI

    app = QApplication(sys.argv)
    game = GameUI()    
    sys.exit(app.exec_())
