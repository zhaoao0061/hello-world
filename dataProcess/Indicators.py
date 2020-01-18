'''

技术指标相关数据处理

'''


import tushare as ts
import pandas as pd
import numpy as np


##移动平均
def MA(close, T):
	if len(close) >= T or type(close):
		return np.sum(close[-T:]) / T
	else:
		print('ERROR in MA, Period is out the data!')
		return 0


##ZT指标
def ZT838(close, N1=10, N2=10):
	a = []
	max_a = []
	min_a = []
	K8 = []
	for i in range(1, len(close)):
		ref_close = close[i - 1]
		single_a = (close[i] - ref_close) / ref_close * 100
		a.append(single_a)
		max_a.append(max(single_a, 0))
		min_a.append(min(single_a, 0))
		if i >= N1:
			K = MA(a, N1)
			D = MA(max_a, N2)
			J = MA(min_a, N2)
			K8.append(1 / (D - J) * (K - J))
	return K8


# 转折点
def FLATZIG(close, rate=0.2):
	position = []
	start = 1
	high_index = 0  # [index,close]
	low_index = 0
	state = 0.5
	down_rate = rate / (1 + rate)

	for i in range(1, len(close)):
		if start == 1:
			if ((close[i] - close[0]) / close[0] > rate):
				high_index = i  # [记录最高点的序号]
				state = 1  # 上涨超过rate，标记为上涨状态
				start = 0
				position.append([0, 0])  # [index,position];position: 0:低点，1：高点
			#					   每次反转确定时，记录上一序号与标识。
			elif close[i] > close[high_index]:
				high_index = i
			if (close[i] - close[0]) / close[0] < -down_rate:
				low_index = i  # 记录最低点的序号
				state = 0  # 下跌超过rate，标记为下跌状态
				start = 0
				position.append([0, 1])
			elif close[i] < close[high_index]:
				low_index = i
		if state == 1:  # 初始为0.5，状态确定为上涨状态后
			if close[i] >= close[high_index]:  # 继续上涨，更新标号
				high_index = i
			if (close[i] - close[high_index]) / close[high_index] < -down_rate:  # 出现了超过阈值的跌幅
				low_index = i
				position.append([high_index, 1])
				state = 0
		if state == 0:  # 状态确定为下跌状态后
			if close[i] <= close[low_index]:
				low_index = i
			if (close[i] - close[low_index]) / close[low_index] > rate:
				high_index = i
				position.append([low_index, 0])
				state = 1
	pos = []
	for i in range(len(position) - 1):
		# position[i][0]:第一列，记录此极点的索引序号。
		# position[i][1]:第二列，记录此点为最高点还是最低点。
		if position[i][1] == 0 and position[i + 1][1] == 1:  # 当前为极小值，下一点为极大值。
			low = close[position[i][0]]
			high = close[position[i + 1][0]]
			div = high - low
			for j in range(position[i][0], position[i + 1][0]):  # 0--61  87--105
				pos.append((close[j] - low) / div)

		elif position[i][1] == 1 and position[i + 1][1] == 0:  # i=62, i+1 = 87   当前为极大值，下一点为极小值。
			high = close[position[i][0]]
			low = close[position[i + 1][0]]
			div = high - low
			for j in range(position[i][0], position[i + 1][0]):  # 62--86
				pos.append((close[j] - low) / div)
		else:
			print('there may be some error,please exam the data of position')
	i += 1
	pos.append(position[i][1])
	# nor_zig_price(position)
	return pos


##KDJ指标
def KDJ(close, low, high, period=27,W1 = 5, W2 = 3):
	trend = []
	CH = []
	sma_1 = []
	for i in range(period - 1, len(close)):
		start = i - period + 1
		if start < 0: start = 0
		CL = close[i] - min(low[start:i + 1])  # 收盘价减去27天最低价的最低值
		HL = max(high[start:i + 1]) - min(low[start:i + 1])
		if HL == 0:
			HL = 0.0001
		CH.append(CL / HL * 100)
	sma_1 = np.array(SMA(CH, W1, 1))  # sma_1: list
	sma_2 = np.array(SMA(sma_1, W2, 1))
	trend = sma_1 * 3 - sma_2 * 2

	return trend / 100


def SMA(price, N, weight):
	sma = []
	for i in range(len(price)):
		if i == 0:
			sma.append(price[i])
		else:
			sma.append((weight * price[i] + (N - weight) * sma[i - 1]) / N)
	return sma
