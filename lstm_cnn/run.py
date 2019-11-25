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
import lstm_cnn.Evaluate as eva
import lstm_cnn.DataLoad as dtload

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu':0})))

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
	wmae = eva.WMAE(predict_pos,true)
	eval_result = eva.evalute_result(predict_pos,y_pos)


	if is_save:
		result_sum = 'epoch:'+str(sum_epoch)+' loss:'+str(new_loss)+' evaluate:'+str(eval_result)+' code:'+str(code) + '\n'
		eval_save = open('result.txt','a')
		eval_save.write(result_sum)
		eval_save.close()






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
	if True:
		dataLoad = dtload.DataLoad();
		file_name= path + 'pos_'+str(pos_range)+'_'  #pos_40_train_z.npz
		(result, no_pos), (cls_train, pos_train)= dataLoad.data_save(train_mode,file_name=file_name,pos_range=pos_range)
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











