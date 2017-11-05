# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 21:03:37 2017

@author: Edward
"""
from scipy.misc import imsave
import numpy as np
from PIL import Image
import requests

def try_and_solve(tweet):
    
    maze_url = tweet['entities']['media'][0]['media_url']
    return solve_maze(maze_url)






def solve_maze(url):
        
    maze_image = np.array(Image.open(requests.get(url, stream=True).raw))
    
    maze_image = maze_image[15:716,19:721,0] < 200
    maze_image = np.vstack([maze_image[0:601,:],maze_image[600:,:]])
    # I create a boolean array to select off the cells in the maze
    logical = np.zeros((maze_image.shape[1]))
    logical[0] = 1
    logical[11::10] = 1
    logical[6::10] = 1
    
    # create the reduced size image
    reduced_size = maze_image[:,logical>0.0]
    reduced_size = reduced_size[logical>0.0,:]
    
    
    def create_node(mini_mat,i,j):
        # given a 3x3 matrix this function creates graph edges to the neighbours
        neighbours = []
        
        if mini_mat[0,1] == False:
            neighbours.append((i-1,j))
        if mini_mat[1,0] == False:
            neighbours.append((i,j-1))    
        if mini_mat[2,1] == False:
            neighbours.append((i+1,j))    
        if mini_mat[1,2] == False:
            neighbours.append((i,j+1))    
        
        return neighbours
    
    # go through all the cells in the maze to create the graph
    nodes = {}
    for i in range(0,70):
        for j in range(0,70):
            mini = reduced_size[2*i:(2*i)+3,2*j:(2*j)+3]
            nodes[(i,j)] = (create_node(mini, i , j),(i,j))
    
    # start is top left, end is bottom right
    start = (0,0)        
    target = (69,69)
    
    # use a priority queue to graph seach with Dijkstra 
    from queue import PriorityQueue
    q = PriorityQueue()
    
    q.put((0,nodes[(0,0)]))
    
    visited = set([(0,0)])
    parents = {}
    
    searching = True
    while searching and not q.empty():
        distance, (children, parent) = q.get()
        #print(distance,children)
        for child in children:
            if child == target:
                #print('Found target !')
                parents[child] = parent     
                searching = False
            else:
                if child in nodes and child not in visited:
                    q.put((distance+1,(nodes[child])))
                    parents[child] = parent               
                    visited.add(child)
                    
    # Get the path
    current = target
    path = []
    while True:
        path.append(current)
        if current == start:
            break
        current = parents[current]
        
            
    #original_image = imread('DNs_2URW4AcizsC.jpg')[:,:,0] < 200
    #original_image.dtype = 'int8'
    
    # update the reduced image with the path
    reduced_size.dtype = 'int8'
    reduced_size = 1- reduced_size
    for i in range(len(path)-1):
        (start_i, start_j) = path[i]
        (end_i, end_j) = path[i+1]
        
        start_i, end_i = min(start_i, end_i), max(start_i, end_i)
        start_j, end_j = min(start_j, end_j), max(start_j, end_j)
        
        reduced_size[(2*start_i)+1:(2*end_i)+2,(2*start_j)+1:(2*end_j)+2] = 2
        
    base = url.split('/')[-1]
    imsave('solved_mazes/'+base[:-4]+'sol.jpg', reduced_size) # This is a smaller version of original image but you get the idea
    return base[:-4]+'sol.jpg'
    
    
    