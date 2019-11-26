"""
生成直接可用的训练数据、测试数据
"""

import numpy as np


POS = 'pos'
CLS = 'cls'


def map_to_train_cls(cls_train):

	X_train = cls_train['train_x']
	y_train = cls_train['train_y']
	X_test = cls_train['test_x']
	y_test = cls_train['test_y']
	test_x_ori = cls_train['test_x_ori']
	test_y_ori = cls_train['test_y_ori']
	train_x_ori = cls_train['train_x_ori']
	train_y_ori = cls_train['train_y_ori']

	print('X_train shape:', X_train.shape)  # (3709L, 50L, 1L)
	print('y_train shape:', y_train.shape)  # (3709L,)
	print('X_test shape:', X_test.shape)  # (412L, 50L, 1L)
	print('y_test shape:', y_test.shape)  # (412L,)
	print('test_x_ori shape:', test_x_ori.shape)  # (3709L, 50L, 1L)
	print('test_y_ori shape:', test_y_ori.shape)  # (3709L,)
	print('train_x_ori shape:', train_x_ori.shape)  # (412L, 50L, 1L)
	print('train_y_ori shape:', train_y_ori.shape)  # (412L,)

	return (X_train,y_train,X_test,y_test),(test_x_ori,test_y_ori,train_x_ori,train_y_ori)


def map_to_train_pos(pos_train):

	X_train = pos_train['train_x']
	y_train = pos_train['train_y']
	X_test = pos_train['test_x']
	y_test = pos_train['test_y']
	test_x_ori = pos_train['test_x_ori']
	test_y_ori = pos_train['test_y_ori']
	train_x_ori = pos_train['train_x_ori']
	train_y_ori = pos_train['train_y_ori']

	print('X_train shape:', X_train.shape)  # (3709L, 50L, 1L)
	print('y_train shape:', y_train.shape)  # (3709L,)
	print('X_test shape:', X_test.shape)  # (412L, 50L, 1L)
	print('y_test shape:', y_test.shape)  # (412L,)
	print('test_x_ori shape:', test_x_ori.shape)  # (3709L, 50L, 1L)
	print('test_y_ori shape:', test_y_ori.shape)  # (3709L,)
	print('train_x_ori shape:', train_x_ori.shape)  # (412L, 50L, 1L)
	print('train_y_ori shape:', train_y_ori.shape)  # (412L,)

	return (X_train, y_train, X_test, y_test), (test_x_ori, test_y_ori, train_x_ori, train_y_ori)


def get_train_data(path):
	pos_range = 0.2
	train_mode = POS

	if False:
		file_name = path + 'pos_' + str(pos_range) + '_'  # pos_40_train_z.npz
		(result, no_pos), (cls_train, pos_train) = data_save(train_mode, file_name=file_name, pos_range=pos_range)
	# filename='pos_40_' : 转折幅度为40%的训练数据
	# (result, no_pos), (cls_train, pos_train) = getStockData.dataFrameToTrain('002594')
	else:
		file_name = 'pos_' + str(pos_range) + '_'
		if train_mode == CLS:
			cls_train = np.load(path + file_name + 'train_z.npz')
			(X_train, y_train, X_test, y_test), \
			(test_x_ori, test_y_ori, train_x_ori, train_y_ori) \
				= map_to_train_cls(cls_train)
		if train_mode == POS:
			pos_train = np.load(path + file_name + 'train_pos_z.npz')
			(X_train, y_train, X_test, y_test), \
			(test_x_ori, test_y_ori, train_x_ori, train_y_ori) \
				= map_to_train_pos(pos_train)
