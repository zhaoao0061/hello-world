import tushare as ts
import pandas as pd
import numpy as np


def normalise_windows(window_data):
	normalised_data = []
	for window in window_data:
		try:
			normalised_window = [((float(p) - float(window[0]))/ float(window[0]))/2 for p in window]
			normalised_data.append(normalised_window)
		except:
			print('ca')
	return  normalised_data

def dynamic_normalise_close(window_data):
	normalised_data = []
	normalised_cls = []
	for window in window_data:
		try:
			normalised_window = [(((float(p) - float(window[0]))/ float(window[0]))) for p in window]
			normalised_data.append(normalised_window)
		except:
			print('ca')
	max_ncls = max(normalised_data)
	min_ncls = min(normalised_data)
	bias = (max_ncls + min_ncls) / 2
	for cls in normalised_data:
		normalised_cls.append(cls - bias)
	return  normalised_cls


def MA(close,T):
	if len(close) >= T or type(close):
		return np.sum(close[-T:]) / T
	else:
		print('ERROR in MA, Period is out the data!')
		return 0

def ZT838(close, N1 = 10, N2 = 10):
	a = []
	max_a = []
	min_a = []
	K8 = []
	for i in range(1,len(close)):
		ref_close = close[i-1]
		single_a = (close[i] - ref_close) / ref_close * 100
		a.append(single_a)
		max_a.append(max(single_a,0))
		min_a.append(min(single_a,0))
		if i >= N1:
			K = MA(a,N1)
			D = MA(max_a, N2)
			J = MA(min_a, N2)
			K8.append(1/(D-J) * (K-J))
	return K8


def FLATZIG(close,rate=0.2):
	position = []
	start = 1
	high_index = 0 #[index,close]
	low_index = 0
	state = 0.5
	down_rate = rate/(1+rate)

	for i in range(1,len(close)):
		if start == 1:
			if((close[i]-close[0])/close[0] > rate):
				high_index = i   #[记录最高点的序号]
				state = 1  #上涨超过rate，标记为上涨状态
				start = 0
				position.append([0,0])#[index,position];position: 0:低点，1：高点
				#					   每次反转确定时，记录上一序号与标识。
			elif close[i] > close[high_index]:
				high_index = i
			if (close[i] - close[0])/close[0] < -down_rate:
				low_index = i#记录最低点的序号
				state = 0#下跌超过rate，标记为下跌状态
				start = 0
				position.append([0,1])
			elif close[i] <close[high_index]:
				high_index = i
		if state == 1: #初始为0.5，状态确定为上涨状态后
			if close[i] >= close[high_index]:#继续上涨，更新标号
				high_index = i
			if (close[i] - close[high_index]) / close[high_index] < -down_rate: #出现了超过阈值的跌幅
				low_index = i
				position.append([high_index,1])
				state = 0
		if state == 0: #状态确定为下跌状态后
			if close[i] <= close[low_index]:
				low_index = i
			if (close[i] - close[low_index])/close[low_index] > rate:
				high_index = i
				position.append([low_index,0])
				state = 1
	pos = []
	for i in range(len(position)-1):
		# position[i][0]:第一列，记录此极点的索引序号。
		# position[i][1]:第二列，记录此点为最高点还是最低点。
		if position[i][1] == 0 and position[i+1][1] == 1:  #当前为极小值，下一点为极大值。
			low = close[position[i][0]]
			high = close[position[i+1][0]]
			div = high - low
			for j in range(position[i][0],position[i+1][0]): #0--61  87--105
				pos.append((close[j] - low)/ div)

		elif position[i][1] == 1 and position[i+1][1] == 0:#i=62, i+1 = 87   当前为极大值，下一点为极小值。
			high = close[position[i][0]]
			low = close[position[i+1][0]]
			div = high - low
			for j in range(position[i][0],position[i+1][0]):#62--86
				pos.append((close[j] - low)/ div)
		else:
			print('there may be some error,please exam the data of position')
	i+=1
	pos.append(position[i][1])
		#nor_zig_price(position)
	return pos
def BDCZ(close,low,high,period = 27):
	trend = []
	CH = []
	sma_1 = []
	for i in range(period,len(close)):
		start = i - period
		if start < 0: start = 0
		CL = close[i] - min(low[start:i]) #收盘价减去27天最低价的最低值
		HL = max(high[start:i]) - min(low[start:i])
		CH.append(CL/HL*100)
	sma_1 = np.array(SMA(CH,5,1)) #sma_1: list
	sma_2 = np.array(SMA(sma_1,3,1))
	trend = sma_1 * 3 - sma_2 * 2

	# if i >= (period + 4):
	# 	sma_1.append(SMA(CH, 5, 1)[-1] * 3)
	# 	sma1 = SMA(CH, 5, 1)[-1] * 3
	# 	if i > period + 4 + 2:
	# 		sma_2 = SMA(sma_1, 3, 1)[-1] * 2
	# 		trend.append(sma1 - sma_2)
	return trend/100

def SMA(price, N, weight):
	sma = []
	for i in range(len(price)):
		if i == 0:
			sma.append(price[i])
		else:
			sma.append((weight * price[i] + (N - weight) * sma[i-1]) / N)
	return sma

def _nor_volume(volume):#标准化成交量
	nor_vol=[]
	for i in range(len(volume)):
		min_vol = 0
		if i < 50:
			max_vol = max(volume[:50])
			nor_vol.append(volume[i] / (max_vol - min_vol))
		else:
			max_vol = max(volume[:i])
			nor_vol.append(volume[i] / (max_vol - min_vol))
	return nor_vol


def get_test_data(code = '002594', seq_len=100, pos_range=0.15, ktype='W'):#获取一只股票的历史数据进行测试
	priceWeek = ts.get_k_data(code, ktype = ktype)
	year = int(priceWeek['date'][0][:4])
	low = priceWeek['low'][0:].tolist()
	high = priceWeek['high'][0:].tolist()
	open = priceWeek['open'][0:].tolist()
	close = priceWeek['close'][0:].tolist()  # 单独拉取收盘价成交量
	volume = priceWeek['volume'][0:].tolist()
	nor_vol = _nor_volume(volume)  # 标准化成交量
	K8 = ZT838(close, N1=10, N2=10)  # 计算指标数值K8
	position = FLATZIG(close, pos_range)
	bdcz = BDCZ(close,low,high).tolist()
	bdcz_99 = BDCZ(close,low,high,99).tolist()

	# features : close, nor_vol, K8;
	# Label: close; position

	# len_div = len(close) - len(K8)
	# close[:len_div] = []  # 从收盘价中去除K8无值的时间点
	# nor_vol[:len_div] = []
	# position[:len_div] = []

	cls_bd_len_div = len(close) - len(bdcz_99)
	k8_bd_len_div = len(K8) - len(bdcz_99)
	bdcz_len_div = len(bdcz) - len(bdcz_99)
	data_len = [len(close), len(K8), len(bdcz),len(bdcz_99)]
	close[:cls_bd_len_div]=[]#从收盘价中去除K8无值的时间点
	nor_vol[:cls_bd_len_div]=[]
	position[:cls_bd_len_div]=[]
	K8[:k8_bd_len_div] = []
	bdcz[:bdcz_len_div] = []


	pos_x = []
	result_temp = []
	sequence_length = seq_len  #100 target的起始位置

	cls = np.array(close)
	pos = np.array(position)

	for index in range(len(close) - sequence_length + 1):
		result_temp.append(close[index:index + sequence_length])#+1:加入最新的一天，以便预测。 0:100
		result_temp.append(nor_vol[index:index + sequence_length])
		result_temp.append(K8[index:index + sequence_length])
		result_temp.append(bdcz[index:index + sequence_length])
		result_temp.append(bdcz_99[index:index + sequence_length])
		pos_x.append(result_temp)
		result_temp = []
	pos_x = np.array(pos_x)
	pos_x = pos_x.swapaxes(1, 2)
	pos_x[0:, 0:, 0] = normalise_windows(pos_x[:, :, 0])
	pos_target = pos[sequence_length-1:]

	return pos_x,pos_target,cls[sequence_length-1:]


def dataFrameToTrain(code,seq_len = 100,pos_range=0.15):
	if not 'code' in locals().keys():
		code='000538'
		print('code is not defined,default set to 600519')

	priceWeek = ts.get_k_data(code,ktype='W')
	year = int(priceWeek['date'][0][:4])
###################筛选时间
	if year > 2016:#去除新股与次新股
		result = []
		no_pos = []
		cls_train = []
		pos_train = []
		return (result, no_pos), (cls_train, pos_train)
##################################
	close = priceWeek['close'][0:].tolist()#单独拉取收盘价成交量
	low = priceWeek['low'][0:].tolist()
	high = priceWeek['high'][0:].tolist()
	open = priceWeek['open'][0:].tolist()
	volume = priceWeek['volume'][0:].tolist()
	nor_vol = _nor_volume(volume) #标准化成交量
	K8 = ZT838(close,N1=10,N2=10)#计算指标数值K8
	position = FLATZIG(close,pos_range)
	bdcz = BDCZ(close,low,high).tolist()
	bdcz_99 = BDCZ(close,low,high,99).tolist()

#features : close, nor_vol, K8;
#Label: close; position

	cls_bd_len_div = len(close) - len(bdcz_99)
	k8_bd_len_div = len(K8) - len(bdcz_99)
	bdcz_len_div = len(bdcz) - len(bdcz_99)
	data_len = [len(close), len(K8), len(bdcz),len(bdcz_99)]
	close[:cls_bd_len_div]=[]#从收盘价中去除K8无值的时间点
	nor_vol[:cls_bd_len_div]=[]
	position[:cls_bd_len_div]=[]
	K8[:k8_bd_len_div] = []
	bdcz[:bdcz_len_div] = []

	no_pos = []
	result_temp = []
	sequence_length = seq_len + 2
	result = []
	cls=[]

	for index in range(len(close) - sequence_length + 1):
		result_temp.append(close[index:index + sequence_length])
		result_temp.append(nor_vol[index:index + sequence_length])
		result_temp.append(K8[index:index + sequence_length])
		result_temp.append(bdcz[index:index + sequence_length])
		result_temp.append(bdcz_99[index:index + sequence_length])
		no_pos.append(result_temp)
		result_temp = []
	no_pos = np.array(no_pos)
	no_pos = no_pos.swapaxes(1, 2)
	#no_pos[0:, 0:, 0] = normalise_windows(no_pos[:, :, 0])  # 对数据进行标准化
	cls_train = get_train(no_pos)


	len_div = len(close) - len(position) #从其他序列中去除pos无值的点
	close[-len_div:]=[]
	nor_vol[-len_div:] = []
	K8[-len_div:] = []
	bdcz[-len_div:] = []
	bdcz_99[-len_div:] = []
	sequence_length = seq_len


	for index in range(len(close) - sequence_length + 1):
		cls.append(close[index:index + sequence_length])        #记录原始价格
		result_temp.append(close[index:index + sequence_length])
		result_temp.append(nor_vol[index:index + sequence_length])
		result_temp.append(K8[index:index + sequence_length])
		result_temp.append(bdcz[index:index + sequence_length])
		result_temp.append(bdcz_99[index:index + sequence_length])
		result_temp.append(position[index:index + sequence_length])
		result.append(result_temp)
		result_temp = []

	result = np.array(result)
	result = result.swapaxes(1, 2)
	#result[0:,0:,0] = normalise_windows(result[:,:,0])  # 对数据进行标准化

	pos_train = get_pos_train(result)

	return (result,no_pos),(cls_train,pos_train)


def get_pos_train(result): #获取用于预测转折点的训练数据
	row = round(0.9 * result.shape[0])  # 获取训练测试比，行数
	nor_result = result.copy()
	nor_result[0:, 0:, 0] = normalise_windows(result[:, :, 0])  #对收盘价进行标准化

	train = nor_result[:int(row),:,:-1].copy()
	target = result[:int(row),:,-1].copy() # 训练target，最后一列的position
	test = nor_result[int(row):, :, :-1].copy()
	test_target = result[int(row):,:,-1].copy()

	train_x_pos = train
	train_y_pos = target[:,-1]
	test_x_pos = test
	test_y_pos = test_target[:,-1]

	train_x_cls_ori = result[:int(row),:,:-1].copy()
	test_y_cls_ori = test_target[:,-1]
	test_x_cls_ori = result[int(row):, :, :-1].copy()
	train_y_cls_ori = target[:,-1]

	train_data={"train_x":train_x_pos,
				"train_y":train_y_pos,
				"test_x": test_x_pos,
				"test_y": test_y_pos,
				"test_x_ori": test_x_cls_ori,
				"test_y_ori": test_y_cls_ori,
				"train_x_ori": train_x_cls_ori,
				"train_y_ori": train_y_cls_ori}
	return  train_data



def get_train(result):#获取训练与测试数据
	row = round(0.95 * result.shape[0])  # 获取训练测试比，行数
	nor_result = result.copy()
	nor_result[0:,0:,0] = normalise_windows(result[:,:,0])   #对收盘价进行标准化

	train = nor_result[:int(row), :, :].copy() #训练集
	train_x_cls = train[:, :-2, :]  #预测下一个价格，next close。  feature：100个cls、K8、volume。 shape:[?,100,3]
	train_y_cls = train[:, -2:, 0]  #标签，序列后两个时间点的close值。[?,2]
	test_x_cls = nor_result[int(row):, :-2, :]
	test_y_cls = nor_result[int(row):, -2:, 0]

	test_x_cls_ori = result[int(row):, :-2, 0].copy() #测试序列的原始价格
	test_y_cls_ori = result[int(row):, -2:, 0].copy()
	train_x_cls_ori = result[:int(row), :-2, 0].copy()
	train_y_cls_ori = result[:int(row), -2:, 0].copy()
	#shape:[total number, sequence lence, feature x number]:[n,100,3]

	train_data={"train_x":train_x_cls,
				"train_y":train_y_cls,
				"test_x": test_x_cls,
				"test_y": test_y_cls,
				"test_x_ori": test_x_cls_ori,
				"test_y_ori": test_y_cls_ori,
				"train_x_ori": train_x_cls_ori,
				"train_y_ori": train_y_cls_ori}

	return train_data



def getStockCode(type=None):
	code = []
	sz50 = pd.read_csv('sz50_code.csv')
	sz50 = sz50['code50'].tolist()

	for i in range(600000,600140):#600140
		code.append(str(i))
	remove_list = ['600026','600053','600083','600090','600132']
	for remove_code in remove_list:
		code.remove(remove_code)
	code.remove('600001')
	code.remove('600002')
	code.remove('600003')
	code.remove('600005')
	code.remove('600013')
	code.remove('600014')
	code.remove('600023')
	code.remove('600024')
	code.remove('600025')
	code.remove('600032')
	code.remove('600034')
	code.remove('600040')
	code.remove('600041')
	code.remove('600042')
	code.remove('600043')
	code.remove('600044')
	code.remove('600045')
	code.remove('600046')
	code.remove('600047')
	code.remove('600049')
	code.remove('600065')
	code.remove('600074')
	code.remove('600087')
	code.remove('600091')
	code.remove('600102')
	code.remove('600124')

	return code
