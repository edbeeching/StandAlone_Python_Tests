# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:50:14 2017

@author: Edward
"""
import random
import numpy as np
from keras import Model
from keras.layers import Dense, Input
from keras.utils import to_categorical

class MovementPolicy():
    
    def __init__(self, random_policy=False):
        self.random_policy = random_policy
        self.agent = self.create_model()
        self.X = None
        self.Y = None
             
    def create_model(self):       
        model_input = Input(shape=(9,))
        x = Dense(12)(model_input)
        model_output = Dense(9, activation='softmax')(x)
        
        model = Model(model_input, model_output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model
     
    @staticmethod
    def get_permutations(board, y):
        boards = [board]
        print(y.shape)
        moves = [y]
        square = board.reshape((3,3))
        move_square = np.zeros((1,9))
        print(move_square, y)
        move_square[0, y[0]] = 1.0
        move_square = move_square.reshape((3, 3))
        
        # compute the 3 other rotations
        for i in range(4):
            boards.append(np.rot90(square, i+1).reshape(1,9))
            moves.append(np.argmax(np.rot90(move_square, i+1).reshape(1,9)))
        
        boards.append(square[:,::-1].reshape(1,9))
        moves.append(np.argmax(move_square[:,::-1].reshape(1,9)))  
        
        boards.append(np.rot90(square[:,::-1]).reshape(1,9))
        moves.append(np.argmax(np.rot90(move_square[:,::-1]).reshape(1,9)))         
        
        return np.vstack(boards), np.vstack(moves)
        
        
         
    def train_model(self):
        if self.X is None or self.Y is None: return
        tmpX = []
        tmpY = []
        
        for board, move in zip(self.X, self.Y):
            Xp, Yp = MovementPolicy.get_permutations(board, move)
            print(Xp.shape, Yp.shape)
            tmpX.append(Xp)
            tmpY.append(Yp)
        
        tmpX = np.vstack(tmpX)
        tmpY = np.vstack(tmpY)
        print('train shapes ', tmpX.shape, tmpY.shape)
        
        print('Updating Model Weights')
        self.agent.fit(self.X, to_categorical(self.Y, num_classes=9), batch_size=16, epochs=500, verbose=0)
        print('Model is trained')
        
    def get_next_move(self, game_board, player):
        if self.random_policy:
            return MovementPolicy.get_random_move(game_board)
        else:
            return self.custom_policy(game_board, player)
                
    @staticmethod
    def get_random_move(game_board):
        if 0.0 not in game_board: assert(0)
            
        move_i = np.random.randint(0,3)
        move_j = np.random.randint(0,3)
        while not game_board[move_i, move_j] == 0.0:
            move_i = np.random.randint(0,3)
            move_j = np.random.randint(0,3)
        return move_i, move_j
        
    def custom_policy(self, game_board, player):
        game_board = game_board.flatten()*player
        game_mask = game_board == 0.0 # Identify where I can play
        print(game_board, player)
        move_probs = self.agent.predict(game_board.reshape((1,9)), verbose=1)*game_mask
        
        best_move = np.argmax(move_probs)
        print('Best Move is ', best_move)
        move_i = best_move // 3
        move_j = best_move % 3
        
        return move_i, move_j
        
    
    def update(self,history, winner):
        if winner == 0:
            self.update(history, 1)
            self.update(history, -1)
            return
    
        states = [board.flatten()*player for (board, player), (i,j) in history if player == winner]
        moves = [i*3+j for (board, player), (i,j) in history if player == winner]

        if self.X is None:
            self.X = np.vstack(states)
        else:
            self.X = np.vstack([self.X, np.vstack(states)])
        
        if self.Y is None:
            self.Y = np.array(moves).reshape((-1,1))
        else:
            self.Y = np.vstack([self.Y, np.array(moves).reshape((-1,1))])

        states = [board.flatten()*player for (board, player), (i,j) in history if player != winner]
        moves = [random.choice([v for v in np.where(board == 0)[0].tolist() if v != (i*3)+j]) for (board, player), (i,j) in history if player != winner]

        self.X = np.vstack([self.X, np.vstack(states)])
        self.Y = np.vstack([self.Y, np.array(moves).reshape((-1,1))])
        print('X ', self.X.shape, ' Y ', self.Y.shape )
        print(self.X)
        print('####################')
        print(self.Y)
        print('####################')
              
        self.train_model()
    
    