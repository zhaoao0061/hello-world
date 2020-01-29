"""
生成直接可用的训练数据、测试数据
"""

import numpy as np
import dataProcess.DataPreProcess as getData


POS = 'pos'
CLS = 'cls'

pos_range = 0.2
train_model = POS


def map_to_train(train_dict):

	X_train = train_dict['train_x']
	y_train = train_dict['train_y']
	X_test = train_dict['test_x']
	y_test = train_dict['test_y']
	test_x_ori = train_dict['test_x_ori']
	test_y_ori = train_dict['test_y_ori']
	train_x_ori = train_dict['train_x_ori']
	train_y_ori = train_dict['train_y_ori']

	print('X_train shape:', X_train.shape)  # (3709L, 50L, 1L)
	print('y_train shape:', y_train.shape)  # (3709L,)
	print('X_test shape:', X_test.shape)  # (412L, 50L, 1L)
	print('y_test shape:', y_test.shape)  # (412L,)
	print('test_x_ori shape:', test_x_ori.shape)  # (3709L, 50L, 1L)
	print('test_y_ori shape:', test_y_ori.shape)  # (3709L,)
	print('train_x_ori shape:', train_x_ori.shape)  # (412L, 50L, 1L)
	print('train_y_ori shape:', train_y_ori.shape)  # (412L,)

	return (X_train,y_train,X_test,y_test),(test_x_ori,test_y_ori,train_x_ori,train_y_ori)


def data_save(file_name='',pos_range = 0.2):
	stockCode = getData.get_szhl_code()#上证红利成分股
	(result, no_pos), (cls_train, pos_train) = getData.dataFrameToTrain(stockCode[0], seq_len = 100, pos_range = pos_range)
	for i in range(1, len(stockCode)):
		try:
			print(stockCode[i] + " Loading.")
			(result_T, no_pos_T), (cls_train_T, pos_train_T) = getData.dataFrameToTrain(stockCode[i], seq_len = 100, pos_range = pos_range)
		except:
			print('数据下载出错： stockCode: '+ stockCode[i])
			continue
		if result_T == []:
			continue
		for ctr in cls_train:
			cls_train[ctr] = np.r_[cls_train[ctr], cls_train_T[ctr]]
			pos_train[ctr] = np.r_[pos_train[ctr], pos_train_T[ctr]]
		result = np.r_[result, result_T]
		no_pos = np.r_[no_pos, no_pos_T]
	if train_model == 'cls' and len(cls_train) > 0:
		np.savez(file_name+'train_z.npz',
				 train_x=cls_train['train_x'],
				 train_y=cls_train['train_y'],
				 test_x=cls_train['test_x'],
				 test_y=cls_train['test_y'],
				 test_x_ori=cls_train['test_x_ori'],
				 test_y_ori=cls_train['test_y_ori'],
				 train_x_ori=cls_train['train_x_ori'],
				 train_y_ori=cls_train['train_y_ori'],
				 )
	elif train_model == 'pos'and len(pos_train) > 0:
		np.savez(file_name+'train_pos_z.npz',
				 train_x=pos_train['train_x'],
				 train_y=pos_train['train_y'],
				 test_x=pos_train['test_x'],
				 test_y=pos_train['test_y'],
				 test_x_ori=pos_train['test_x_ori'],
				 test_y_ori=pos_train['test_y_ori'],
				 train_x_ori=pos_train['train_x_ori'],
				 train_y_ori=pos_train['train_y_ori'],
				 )
	return (result, no_pos), (cls_train, pos_train)

#从网络下载数据并保存
def get_train_save(path, pos_range = 0.2):

	file_name = path + 'pos_' + str(pos_range) + '_'  # pos_40_train_z.npz
	(result, no_pos), (cls_train, pos_train) = data_save(file_name = file_name, pos_range = pos_range)
	(X_train, y_train, X_test, y_test), (test_x_ori, test_y_ori, train_x_ori, train_y_ori) \
		= map_to_train(pos_train)
	return (X_train, y_train, X_test, y_test)

#从文件加载数据
def get_train_load(file_name):
	pos_train = np.load(file_name)
	(X_train, y_train, X_test, y_test), (test_x_ori, test_y_ori, train_x_ori, train_y_ori) \
		= map_to_train(pos_train)
	return (X_train, y_train, X_test, y_test)
