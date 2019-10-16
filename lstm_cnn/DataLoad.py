import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import getStockData


class DataLoad():
	def __init__(self):
		print("DataLoad Begin")

	def data_save(train_model,file_name='',pos_range = 0.2):
		stockCode = getStockData.getStockCode()
		(result, no_pos), (cls_train, pos_train) = getStockData.dataFrameToTrain(stockCode[0],pos_range=pos_range)
		for i in range(1, len(stockCode)):
			try:
				(result_T, no_pos_T), (cls_train_T, pos_train_T) = getStockData.dataFrameToTrain(stockCode[i],pos_range=pos_range)
			except:
				print('dddd')
				continue
			if result_T == []:
				continue
			for ctr in cls_train:
				cls_train[ctr] = np.r_[cls_train[ctr], cls_train_T[ctr]]
				pos_train[ctr] = np.r_[pos_train[ctr], pos_train_T[ctr]]
			result = np.r_[result, result_T]
			no_pos = np.r_[no_pos, no_pos_T]
		if train_model == 'cls':
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
		elif train_model == 'pos':
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


	def map_to_train(train_mode):
		if train_mode == 'cls':
			X_train = cls_train['train_x']
			y_train = cls_train['train_y']
			X_test = cls_train['test_x']
			y_test = cls_train['test_y']
			test_x_ori = cls_train['test_x_ori']
			test_y_ori = cls_train['test_y_ori']
			train_x_ori = cls_train['train_x_ori']
			train_y_ori = cls_train['train_y_ori']
		elif train_mode == 'pos':
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

		return (X_train,y_train,X_test,y_test),(test_x_ori,test_y_ori,train_x_ori,train_y_ori)
