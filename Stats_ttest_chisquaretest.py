# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:14:28 2020

@author: zhaor
"""
######## t-test ########
from __future__ import print_function, division
from builtins import range
import numpy as np
from scipy import stats

# generate data
N = 10
a = np.random.randn(N) + 2 # mean 2, variance 1
b = np.random.randn(N) # mean 0, variance 1

# build from formula t-test:
var_a = a.var(ddof=1) # unbiased estimator, divide by N-1 instead of N
var_b = b.var(ddof=1)
s = np.sqrt( (var_a + var_b) / 2 ) # balanced standard deviation
t = (a.mean() - b.mean()) / (s * np.sqrt(2.0/N)) # t-statistic
df = 2*N - 2 # degrees of freedom
p = 1 - stats.t.cdf(np.abs(t), df=df) # one-sided test p-value
print("t:\t", t, "p:\t", 2*p) # two-sided test p-value

# built-in t-test:
t2, p2 = stats.ttest_ind(a, b)
print("t2:\t", t2, "p2:\t", p2)

########## chi-square test ##########
from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, chi2_contingency

# contingency table
#        click       no click
#------------------------------
# ad A |   a            b
# ad B |   c            d
#
# chi^2 = (ad - bc)^2 (a + b + c + d) / [ (a + b)(c + d)(a + c)(b + d)]
# degrees of freedom = (#cols - 1) x (#rows - 1) = (2 - 1)(2 - 1) = 1

# short example

# T = np.array([[36, 14], [30, 25]])
# c2 = np.linalg.det(T)**2 * T.sum() / ( T[0].sum()*T[1].sum()*T[:,0].sum()*T[:,1].sum() )
# p_value = 1 - chi2.cdf(x=c2, df=1)

# equivalent:
# (36-31.429)**2/31.429+(14-18.571)**2/18.571 + (30-34.571)**2/34.571 + (25-20.429)**2/20.429


class DataGenerator:
  def __init__(self, p1, p2):
    self.p1 = p1
    self.p2 = p2

  def next(self):
    click1 = 1 if (np.random.random() < self.p1) else 0
    click2 = 1 if (np.random.random() < self.p2) else 0
    return click1, click2


def get_p_value(T):
  # same as scipy.stats.chi2_contingency(T, correction=False)
  det = T[0,0]*T[1,1] - T[0,1]*T[1,0]
  c2 = float(det) / T[0].sum() * det / T[1].sum() * T.sum() / T[:,0].sum() / T[:,1].sum()
  p = 1 - chi2.cdf(x=c2, df=1)
  return p


def run_experiment(p1, p2, N):
  data = DataGenerator(p1, p2)
  p_values = np.empty(N)
  T = np.zeros((2, 2)).astype(np.float32)
  for i in range(N):
    c1, c2 = data.next()
    T[0,c1] += 1
    T[1,c2] += 1
    # ignore the first 10 values
    if i < 10:
      p_values[i] = None
    else:
      p_values[i] = get_p_value(T)
  plt.plot(p_values)
  plt.plot(np.ones(N)*0.05)
  plt.show()

run_experiment(0.1, 0.11, 20000)