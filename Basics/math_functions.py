"""
25 September 2017
@author: Allan Perez
"""
import numpy as np

def linearRegress(w, x):
	return np.dot(w,x)

def logisticRegress(g):
	return (1+np.exp(-g))**-1