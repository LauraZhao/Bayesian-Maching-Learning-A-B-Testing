# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:56:45 2020

@author: zhaor
"""

from __future__ import print_function, division
from builtins import range
import matplotlib.pyplot as plt
import numpy as np
from bayesian_bandit import Bandit      #import the "Bandit" class from the py file we already create


def run_experiment(p1, p2, p3, N):                     #converge experiment function: 
  bandits = [Bandit(p1), Bandit(p2), Bandit(p3)]       #Create 3 bandits with probability p1,p2,p3; we play 100000 times

  data = np.empty(N)                                   #store results in data
  
  for i in range(N):
    # thompson sampling
    j = np.argmax([b.sample() for b in bandits])
    x = bandits[j].pull()
    bandits[j].update(x)

    # for the plot
    data[i] = x                                        #x means "Get a reward or not" or "click or not" 
  cumulative_average_ctr = np.cumsum(data) / (np.arange(N) + 1)#we keep track of the cumulative click_through_rate by
                                                               #divide the total number of click by the total showing time.
  # plot moving average ctr                                    #when we are using Thompson Sampling.
  plt.plot(cumulative_average_ctr)
  plt.plot(np.ones(N)*p1)
  plt.plot(np.ones(N)*p2)
  plt.plot(np.ones(N)*p3)
  plt.ylim((0,1))
  plt.xscale('log')
  plt.show()                                           #What we see here is after 100000 trials, the average ctr 
                                                       #becomes very close to 0.3, which is the highest one.
                                                       #And that's how Thompson Sampling can automatically converge 
                                                       #to the best situation.


run_experiment(0.2, 0.25, 0.3, 100000)