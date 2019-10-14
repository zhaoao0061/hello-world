import lstm_cnn.lstmTimeSeries as lstm
import time
import matplotlib
matplotlib.use('TkAgg')
import math
import matplotlib.pyplot as plt
import numpy as np
import tushare as ts
import getStockData
import lstm_cnn.tempcode as tpc
import pandas as pd
import os
from keras.models import load_model
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras import optimizers,losses

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu':0})))

def plot_results_multiple(predicted_data, true_data, prediction_len):
	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	ax.plot(true_data, label = 'True Data')

	for i, data in enumerate(predicted_data):
		padding = [None for p in range(i * prediction_len)]
		plt.plot(padding + data, label = 'Prediction')
		plt.legend()
	plt.show()

def plot_results(predicted_data, true_data, filename):
	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	ax.plot(true_data, label = 'True Data')
	plt.plot(predicted_data, label='Prediction')
	plt.legend()
	plt.show()
def back_price0(nor_data,ori_data):
	back_nor_data = []
	for (y,oridata) in zip(nor_data,ori_data):
		back_nor = (y*2)*oridata[0]+oridata[0]
		back_nor_data.append(back_nor)
	return back_nor_data

def back_price(y,ori_data):
    back_price = []
    for (prc,oridata) in zip(y,ori_data):
        backPrice = (prc+1)*oridata[-1]
        back_price.append(backPrice)
    return back_price

def pred_test(code,is_save=False,index_num=0,sum_epoch=0,pos_range=0):

	pos_x, pos_target,pos_cls = getStockData.get_test_data(code=code,pos_range=pos_range)
	pos_target = tpc.toTen(pos_target, 101)
	pos_x_cnn = np.reshape(pos_x, [pos_x.shape[0], 1, 100, 5])
	#pos_x_test,y_train = tpc.xTo2D(X_train,y_train,150)

	predict_ten = model.predict([pos_x,pos_x_cnn])
	believe_P = np.max(predict_ten,1) * 100
	predict_pos=[]
	y_pos = []
	for i in range(len(predict_ten)):
		predict_pos.append(np.argmax(predict_ten[i,:]))
		if i < len(pos_target):
			y_pos.append(np.argmax(pos_target[i,:]))
	plt.plot(predict_pos,label='pred')
	plt.plot(y_pos,label='true')
	plt.plot(pos_cls * 3, label='close')
	# plt.plot(believe_P,label='believe')
	plt.legend(loc='upper right')
	plt.rcParams['figure.figsize'] = (16.0, 8.0)
	plt.show()
	plt.rcParams['savefig.dpi'] = 700
	model_name = str(index_num) + '_' + str(sum_epoch) + '_pos_' + str(pos_range) + '_lstm_model'
	plt.savefig('modelfiles/' + model_name + code + '.png', format='png',dpi=700)
	plt.close()

	target_len = len(y_pos)
	pred = predict_pos[:target_len]
	true = y_pos
	wmae = WMAE(predict_pos,true)
	eval_result = evalute_result(predict_pos,y_pos)


	if is_save:
		result_sum = 'epoch:'+str(sum_epoch)+' loss:'+str(new_loss)+' evaluate:'+str(eval_result)+' code:'+str(code) + '\n'
		eval_save = open('result.txt','a')
		eval_save.write(result_sum)
		eval_save.close()

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

def data_save(train_model,file_name='',pos_range = 0.2):
	stockCode = getStockData.getStockCode()
	stockCode = stockCode
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
	elif train_mode == 'pos':
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




if __name__=='__main__':
	global_start_time = time.time()
	epochs = 17
	seq_len = 100
	POS = 'pos'
	CLS = 'cls'

	train_mode = POS

	print('> Loading data... ')
	#nor_result = result
	path = 'datafiles/'
	pos_range=0.4
	if False:
		file_name= path + 'pos_'+str(pos_range)+'_'  #pos_40_train_z.npz
		(result, no_pos), (cls_train, pos_train)=data_save(train_mode,file_name=file_name,pos_range=pos_range)
		#filename='pos_40_' : 转折幅度为40%的训练数据
		#(result, no_pos), (cls_train, pos_train) = getStockData.dataFrameToTrain('002594')
	else:
		file_name = 'pos_'+str(pos_range)+'_'
		if train_mode == CLS: cls_train = np.load(path + file_name + 'train_z.npz')
		if train_mode == POS: pos_train = np.load(path + file_name + 'train_pos_z.npz')
	(X_train, y_train, X_test, y_test), \
	(test_x_ori, test_y_ori, train_x_ori, train_y_ori)\
		= map_to_train(train_mode)
	print('> Data Loaded. Compiling...')

	y_test = tpc.toTen(y_test,101)
	y_train = tpc.toTen(y_train,101)

	#X_train,y_train = tpc.xTo2D(X_train,y_train,150)
	#X_test,y_test = tpc.xTo2D(X_test,y_test,150)
	xcnn_train = np.reshape(X_train, [X_train.shape[0], 1, 100, 5])
	xcnn_test = np.reshape(X_test, [X_test.shape[0], 1, 100, 5])


	start = time.time()
	input_nodes = X_train.shape[2]
	if y_train.ndim == 1:
		output_nodes = 1
	else:
		output_nodes = y_train.shape[1]


	model,model_dr = lstm.share_model3(class_num=101)
	model.summary()

	sum_epoch = 11

	# loss_model = losses.tr_distance_categorical_crossentropy
	#loss_model = losses.mae_categorical_crossentropy
	loss_model = losses.categorical_crossentropy
	# loss_model = losses.mae_dis_categorical_crossentropy
	rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
	model.compile(loss=loss_model, optimizer=rmsprop, metrics=['accuracy'])

	for _ in range(5):
		#pre_loss = new_loss.copy()
		epochs = 2
		hist = model.fit(
			[X_train,xcnn_train],
			y_train,
			batch_size =70,
			nb_epoch = epochs,
			validation_split = 0.4,
			#class_weight={0:2, 1:1.8, 2:1.8, 3:0.6, 4:0.1, 5:0.1, 6:0.1, 7:0.6, 8:1.8, 9:1.8, 10:2}
			#class_weight = {0: 1.8, 1: 1.8, 2: 1.8, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1, 10: 0.1}
			#class_weight={0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 1.8, 9: 1.8, 10: 1.8}
		)
		sum_epoch += epochs
		new_loss = hist.history['loss']

		index_num = 125
		model_path = 'modelfiles2019/'
		model_name=str(index_num)+ '_' + str(sum_epoch) + '_pos_'+str(pos_range)+'_lstm_model' #37_pos_40_lstm_model
		model.save(model_path + model_name+'.h5')
		model.save_weights(model_path + model_name+'_weights.h5')

		# model_name = '2_69_pos_0.6_lstm_model'  # 37_pos_40_lstm_model
		# model_name = '0_minErr_19_pos_0.15_lstm_model'
		# model = load_model('modelfiles/'+ model_name + '.h5')
		# model.load_weights('modelfiles/'+ model_name + '_weights.h5')


		print('fitTime:',time.time()-start)
		#predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)

		#print('Training duration (s) : ', time.time() - global_start_time)
		#plot_results_multiple(predictions, y_test, 49)
		code_list=['002154','000536','600108','601006',]
		#code='002154'
		for code in code_list:
			pos_x, pos_target,pos_cls = getStockData.get_test_data(code=code,pos_range=pos_range)
			pos_target = tpc.toTen(pos_target, 101)
			pos_x_cnn = np.reshape(pos_x, [pos_x.shape[0], 1, 100, 5])
			#pos_x_test,y_train = tpc.xTo2D(X_train,y_train,150)

			predict_ten = model.predict([pos_x,pos_x_cnn])
			believe_P = np.max(predict_ten,1) * 100
			predict_pos=[]
			y_pos = []
			for i in range(len(predict_ten)):
				predict_pos.append(np.argmax(predict_ten[i,:]))
				if i < len(pos_target):
					y_pos.append(np.argmax(pos_target[i,:]))
			plt.plot(predict_pos,label='pred')
			plt.plot(y_pos,label='true')
			plt.plot(pos_cls * 3, label='close')
			# plt.plot(believe_P,label='believe')
			plt.legend(loc='upper right')
			plt.rcParams['figure.figsize'] = (16.0, 8.0)
			plt.show()
			plt.rcParams['savefig.dpi'] = 700
			model_name = str(index_num) + '_' + str(sum_epoch) + '_pos_' + str(pos_range) + '_lstm_model'
			plt.savefig( model_path + model_name + code + '.png', format='png',dpi=700)
			plt.close()

			eval_result = evalute_result(predict_pos,y_pos)
			result_sum = 'epoch:'+str(sum_epoch)+' loss:'+str(new_loss)+' evaluate:'+str(eval_result)+' code:'+str(code) + '\n'
			eval_save = open('result.txt','a')
			eval_save.write(result_sum)
			eval_save.close()


	pre_mean_bottom = evalute_result(predict_pos,y_pos)
	min_mean_bottom = pre_mean_bottom.copy()
	lrs = 0.0001

	for _ in range(5):
		epochs = 1
		hist = model.fit(
			[X_train, xcnn_train],
			y_train,
			batch_size=70,
			nb_epoch=epochs,
			validation_split=0.5,
			# class_weight={0:2, 1:1.8, 2:1.8, 3:0.6, 4:0.1, 5:0.1, 6:0.1, 7:0.6, 8:1.8, 9:1.8, 10:2}
			# class_weight = {0: 1.8, 1: 1.8, 2: 1.8, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1, 10: 0.1}
			# class_weight={0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 1.8, 9: 1.8, 10: 1.8}
		)
		sum_epoch += epochs


		predict_ten = model.predict([pos_x,pos_x_cnn])
		believe_P = np.max(predict_ten, 1) * 100
		predict_pos = []
		for i in range(len(predict_ten)):
			predict_pos.append(np.argmax(predict_ten[i, :]))
		mean_bottom = evalute_result(predict_pos,y_pos)

		print(mean_bottom)

		if mean_bottom < min_mean_bottom:
			model_name = str(index_num) + '_minErr_' + str(sum_epoch) + '_pos_' + str(
				pos_range) + '_lstm_model'  # 37_pos_40_lstm_model
			model.save('modelfiles/' + model_name + '.h5')
			model.save_weights('modelfiles/' + model_name + '_weights.h5')
			plt.close()
			plt.plot(predict_pos, label='pred')
			plt.plot(y_pos, label='true')
			plt.plot(pos_cls * 3,label = 'close')
			plt.plot(believe_P, label='believe')
			plt.legend(loc='upper right')
			plt.rcParams['figure.figsize'] = (16.0, 8.0)
			plt.show()
			plt.rcParams['savefig.dpi'] = 700
			model_name = str(index_num) + '_minErr_' + str(sum_epoch) + '_pos_' + str(pos_range) + '_lstm_model'
			plt.savefig('modelfiles/' + model_name + code + '.png', format='png', dpi=700)
			plt.close()

			min_mean_bottom = mean_bottom.copy()
		elif mean_bottom > pre_mean_bottom:
			lrs *= 0.65
			rmsprop = optimizers.RMSprop(lr=lrs, rho=0.9, epsilon=1e-06)
			# opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
			# loss_model = losses.pos_error
			# loss_model_mae = losses.mean_absolute_error
			model.compile(loss=losses.categorical_crossentropy, optimizer=rmsprop, metrics=['accuracy'])

		pre_mean_bottom = mean_bottom.copy()



sma_pred_15 = getStockData.SMA(predict_pos,15,1)
# ma_pre = getStockData.MA(predict_pos,10)
plt.plot(sma_pred_15,label='sma_pred_15')

five = [5] *len(predict_pos)
plt.plot(five)
eighty = [85] * len(predict_pos)
plt.plot(eighty)











