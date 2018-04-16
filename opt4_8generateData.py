#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
seed=2

def generates():
	rdm=np.random.RandomState(seed)
	X=rdm.randn(300,2)
	Y_=[int(x0*x0+x1*x1<2) for (x0,x1) in X]
	Y_c=[['red' if y else 'blue'] for y in Y_]
	#对数据集进行整理，第一个元素为-1表示跟随第二列计算，第二个元素表示多少     #列
	X=np.vstack(X).reshape(-1,2)
	Y_=np.vstack(Y_).reshape(-1,1)
	return X,Y_,Y_c
