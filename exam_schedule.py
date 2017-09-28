# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 16:38:57 2017

@author: Edward Beeching
"""

import itertools


def scoring_function(times):
    """
        you can score it how you like, but I try to maximise the gaps between exams. but clip values greater than 4 and punish values less than 1
    """
    sorted_times = sorted(times)
    
    diffs = []
    for i in range(len(sorted_times)-1):
        diff = sorted_times[i+1]- sorted_times[i]
        
        if diff == 0.0: # overlaps cannot happen score with a large penalty
            diffs.append(-100)
        elif diff <= 1.0: # punish small differences
            diffs.append(-2)
        elif diff > 4.0: # Gaps greater than 4 are large enough and considered OK
            diffs.append(4.0)
        else:
            diffs.append(diff)
      
    return sum(diffs)

subjects = 'machine_learning  neural_computing genetic_algos speech_recog wavelets cybernetics NLP'.split()

# 1 day = 1.0, 1 day + morning = 1.5. e.g Monday AM to Tues AM = 1.0, Monday AM to Tuesday PM = 1.5
# first exam day starts at 0.0
machine_learning_times = [0.5, 12.0, 17.0]
neural_computing_times = [3.0, 14.0]
genetic_algos_times = [1.0, 3.0, 3.5, 14, 14.5, 20, 20.5]
speech_recog_times = [4.0, 4.5, 12.0]
wavelets_times = [10.0, 17.0]
cybernetics_times = [10.0, 18.0]
nlp_times = [17.5]


best_score = -1
best_times = None

for ml, nc, ga, sr, w, c, npl in itertools.product(machine_learning_times, 
                                                   neural_computing_times,
                                                   genetic_algos_times,
                                                   speech_recog_times,
                                                   wavelets_times,
                                                   cybernetics_times,
                                                   nlp_times):
    cur_score = scoring_function([ml, nc, ga, sr, w, c, npl])
    if cur_score > best_score:
        best_score = cur_score
        best_times = (ml, nc, ga, sr, w, c, npl)
    
    
for subject, time in zip(subjects, best_times):
    print('Subject {}  at time {}'.format(subject, time))


# ML Fri 12 PM
# SPEECH Recog Tue 16 PM
# WAVELET Mon 22 AM
# Neural computing Fri 26 PM
# NLP Mon 29 AM
# Cybernetics Tue 30 PM
# Genetic Algos Thu 1 PM


