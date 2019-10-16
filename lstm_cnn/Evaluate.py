
import matplotlib
matplotlib.use('TkAgg')
import numpy as np



def WMAE(pred,true):
	# yibai = np.array(range(0,100))
	# yibai_2 = np.array(yibai)/50-1
	# c = 0.01
	# wy = np.log((1 + yibai_2 + c) / (1 - yibai_2 + c))
	# plt.plot(wy)
	pred = np.array(pred)/50-1
	true = np.array(true)/50-1
	c=0.1
	Pred = 0.5 * np.log((1 + pred + c) / (1 - pred + c))
	Target = 0.5* np.log((1 + true +c) / (1- true + c))
	wab = np.abs(Pred-Target)
	wsq = np.square(Pred-Target)
	wmae = np.sum(wab)/len(wab)
	return wmae

def Turn_error(pred,target):
	sum_dif = 0
	n = 0.00000001
	dif = 0
	for i in range(len(target)):
		if pred[i] == 0:
			dif = target[i] - pred[i]
			sum_dif += dif
			n += 1
	mean_error = sum_dif / n
	print('mean_error:', mean_error)

def evalute_result(pred,true):
	sum_div = 0
	n = 0.00000001
	div=0
	for i in range(len(true)):
		if pred[i] == 0:
			div = true[i] - pred[i]
			sum_div += div
			n += 1
	mean_error = sum_div / n
	print('mean_error:',mean_error)

	return mean_error
