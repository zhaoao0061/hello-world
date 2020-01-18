"""

数据预处理：数据的组织构建

"""



import tushare as ts
import numpy as np
import dataProcess.Indicators as Indicator
import dataProcess.ScaleForStock as Scale




def get_test_data(code = '002594', seq_len=100, pos_range=0.15, ktype='W'):#获取一只股票的历史数据进行测试
	priceWeek = ts.get_k_data(code, ktype = ktype)
	year = int(priceWeek['date'][0][:4])
	low = priceWeek['low'][0:].tolist()
	high = priceWeek['high'][0:].tolist()
	open = priceWeek['open'][0:].tolist()
	close = priceWeek['close'][0:].tolist()  # 单独拉取收盘价成交量
	volume = priceWeek['volume'][0:].tolist()
	nor_vol = Scale._nor_volume(volume)  # 标准化成交量
	K8 = Indicator.ZT838(close, N1=10, N2=10)  # 计算指标数值K8
	position = Indicator.FLATZIG(close, pos_range)
	bdcz = Indicator.KDJ(close,low,high).tolist()
	bdcz_99 = Indicator.KDJ(close,low,high,99).tolist()

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
	pos_x[0:, 0:, 0] = Scale.normalise_windows(pos_x[:, :, 0])
	pos_target = pos[sequence_length-1:]

	return pos_x,pos_target,cls[sequence_length-1:]

#从互联网下载股票数据
def dataFrameToTrain(code, seq_len, pos_range):
	if not 'code' in locals().keys():
		code='000538'
		print('code is not defined,default set to 600519')

	priceWeek = ts.get_k_data(code,ktype='D',start='2010-01-01', end='2020-01-17')
	year = int(priceWeek['date'][0][:4])
###################筛选时间
	if year > 2018:#去除新股与次新股
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
	nor_vol = Scale._nor_volume(volume) #标准化成交量
	K8 = Indicator.ZT838(close,N1=10,N2=10)#计算指标数值K8
	position = Indicator.FLATZIG(close,pos_range)
	bdcz = Indicator.KDJ(close,low,high).tolist()
	bdcz_99 = Indicator.KDJ(close,low,high,100,15,5).tolist()

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
	nor_result[0:, 0:, 0] = Scale.normalise_windows(result[:, :, 0])  #对收盘价进行标准化

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
	nor_result[0:,0:,0] = Scale.normalise_windows(result[:,:,0])   #对收盘价进行标准化

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


def get_szhl_code():
	code = []
	code.append("600011")
	code.append("600019")
	code.append("600023")
	code.append("600028")
	code.append("600033")
	code.append("600036")
	code.append("600048")
	code.append("600066")
	code.append("600104")
	code.append("600162")
	code.append("600177")
	code.append("600183")
	code.append("600269")
	code.append("601006")
	code.append("601009")
	code.append("601088")
	code.append("601158")
	return code
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