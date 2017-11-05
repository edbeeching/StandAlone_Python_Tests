# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 19:35:49 2017

@author: Edward
"""

import os
from twython import Twython
from maze_bot_utils import try_and_solve
import time

keys = []
with open('twitter_keys.txt','r') as f:
    for line in f.readlines():
        keys.append(line)
        
consumer_key, consumer_secret = keys[0][:-1], keys[1][:-1]
access_token, access_token_secret = keys[2][:-1], keys[3][:-1]

while True:
    try:
        twitter = Twython(consumer_key, consumer_secret, access_token, access_token_secret)

        settings = twitter.get_account_settings()
        user_timeline = twitter.get_home_timeline()
        
        solved_mazes = os.listdir('solved_mazes')
        solved_mazes = set([f[:-7]+'.jpg' for f in solved_mazes])
        user_timeline = twitter.get_home_timeline()
        
        for tweet in user_timeline:
            if "Today's maze" in tweet['text']:
                try:   
                    maze_jpgname = tweet['entities']['media'][0]['media_url'].split('/')[-1]
                    if maze_jpgname in solved_mazes:
                        print(maze_jpgname, 'is already solved')
                        continue
                    
                    print('trying to solve',tweet['entities']['media'][0]['media_url'])
                    solved_jpgname = try_and_solve(tweet)
                    solved_mazes.add(solved_jpgname[:-7]+'.jpg')
                    print('tweet id', tweet['id_str'])
                    
                    solution = open('solved_mazes/'+solved_jpgname, 'rb')
                    response = twitter.upload_media(media=solution)
                    twitter.update_status(status='Here is the solution to the maze! @mazeaday', in_reply_to_status_id=tweet['id'], media_ids=[response['media_id']])
                    
                except Exception as e:
                    print('Exception while trying to solve maze', e)
    except Exception as e:
        print('Timeline exception',e)
                        
    time.sleep(10.0)
            
