import numpy as np
import tushare as ts

from sklearn import preprocessing


# 保存训练的模型
def mdsave(model, model_path, index_num, sum_epoch):
    model_name = str(index_num) + '_' + str(sum_epoch) + '_pos_' + '_lstm_model'  # 37_pos_40_lstm_model
    model.save(model_path + model_name + '.h5')
    model.save_weights(model_path + model_name + '_weights.h5')

##############

# 标准化成交量
def nor_volume(volume):
	nor_vol = []
	for i in range(len(volume)):
		min_vol = 0
		if i < 50:
			max_vol = max(volume[:50])
			nor_vol.append(volume[i] / (max_vol - min_vol))
		else:
			max_vol = max(volume[:i])
			nor_vol.append(volume[i] / (max_vol - min_vol))
	return nor_vol



# 数据保存
# def data_save(stockCode, seqlen = 100, file_name = '' ):
def data_save(stockCode, file_name=''):

	# 获取以上代码的训练数据，按照时间窗口处理成多个样本，每个样本的特征在函数里设置
	# result, train = dataFrameToTrain(stockCode[0], seq_len = seqlen)

	x,y = dataFrameToTrain(stockCode[0])

	for i in range(1, len(stockCode)):  # 循环获取所有的股票数据
		try:
			# 获取除了标号'600000'的股票ß
			x_Temp, y_Temp = dataFrameToTrain(stockCode[i])


		except:
			continue
		if y_Temp == []:
			continue

		y = np.r_[y, y_Temp]
		x = np.r_[x, x_Temp]

	np.savez(file_name + '.npz',
			 x = x,
			 y = y,
			 )


	return x,y


# 获取训练数据
def map_to_train(x, y):
	x = x,
	y = y,
	return  x,y


# 样本生成函数，输入代码，序列长度，趋势阈值，就可以生成用于训练的数据，包括训练样本和训练目标。
# 注意生成的数据格式和模型需要的相同就行。
# 里面可以再添加特征。
# def dataFrameToTrain(code, seq_len = 100): # code 是股票数据的代码列表 ##
def dataFrameToTrain(code):  # code 是股票数据的代码列表 ##

	if not 'code' in locals().keys():
		code = '000538'
		print('code is not defined,default set to 600519')

	priceDate = ts.get_k_data(code, ktype='W')  # 获取每日的价格 ###########
	year = int(priceDate['date'][0][:4])

	# 新增日期
	# date = int(priceDate['date'][0:].tolist())

	#################################### 筛选时间 ###################################
	if year > 2016: # 去除新股与次新股
		y = []
		x = []
		return x, y
	###############################################################################

	close = priceDate['close'][0:].tolist() # 单独拉取收盘价成交量
	low = priceDate['low'][0:].tolist()         # 每周的最低价
	high = priceDate['high'][0:].tolist()       # 每周的最高价
	open = priceDate['open'][0:].tolist()       # 每周的开盘价
	volume = priceDate['volume'][0:].tolist()   # 每周的成交量

	volume = nor_volume(volume)   				# 标准化成交量

	# 规范化参数
	close = preprocessing.scale(close)
	open = preprocessing.scale(open)
	# volume = preprocessing.scale(volume)
	low = preprocessing.scale(low)
	high = preprocessing.scale(high)

	result_temp = []
	result = []
	cls = []

	x = np.stack((close, open, volume, low, high), axis = 1)
	# x = np.stack((close, open, volume, low, high), axis=1)
	y = close

	return x,y


# 返回一个列表，为训练集的股票代码。
def getStockCode():
	code = []
	for i in range(600000,600050):
		code.append(str(i))
	remove_list = ['600026'] #, '600053', '600083', '600090', '600132'
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
	# code.remove('600065')
	# code.remove('600074')
	# code.remove('600087')
	# code.remove('600091')
	# code.remove('600102')
	# code.remove('600124')

	return code