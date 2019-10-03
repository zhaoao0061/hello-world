import lstm_cnn.lstmTimeSeries as lstm
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import getStockData
import pandas as pd
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


def evalute_result(pred,true):
	sum_div = 0
	n = -0.00000001
	div=0
	for i in range(len(true)):
		if pred[i] == 0:
			div = true[i] - pred[i]
			sum_div += div
			n += 1
	mean_error = sum_div / n
	summean = (100-mean_error)*n / 10
	print('mean_error:',mean_error,summean)

	return mean_error,summean

def profit():
	return 0


def TAC(pred, target):
	right = 0
	sum_pred_turn = 0
	for i in range(len(target)):
		if pred[i] == 0:
			if target[i] == 0:
				right += 1
			sum_pred_turn += 1
	if sum_pred_turn > 0:
		return right/sum_pred_turn*100
	else:
		return -1

def WMAE(pred,true):
	# yibai = np.array(range(0,101))
	# yibai_2 = (np.array(yibai)/100-0.5)*4
	# c = 0
	# ly = np.clip((1 + yibai_2 + c) / (1 - yibai_2 + c),1e-2,100)
	# wy = np.log(ly)
	#wy = yibai_2 ** 3
	# plt.plot(wy)
	pred = np.clip(np.array(pred)/50-1, -0.99,0.99)
	true = np.clip(np.array(true)/50-1, -0.99,0.99)
	pred = pred[:len(true)]
	c=0.001

	Pred = 0.5 * np.log((1 + pred) / (1 - pred))
	Target = 0.5* np.log((1 + true) / (1- true))
	Target = np.reshape(Target,[len(Target),1])
	wab = np.abs(Pred-Target)
	wsq = np.square(Pred-Target)
	wmae = np.sum(wab,keepdims=False)/len(wab)
	return wmae

def Recall(pred,target,below = 5):  # 查全误差
	sum_dif = 0
	n = 0.00000001
	dif = 0
	for i in range(len(target)):
		if target[i] <= below:
			dif = pred[i] - target[i]
			sum_dif += dif
			n += 1
	mean_error = sum_dif / n
	print('Recall:', mean_error)
	if n >= 1:
		return mean_error
	else:
		return [-1]


def Turn_error(pred,target,below = 5):  #查准误差
	sum_dif = 0
	n = 0.00000001
	dif = 0
	for i in range(len(target)):
		if pred[i] <= below:
			dif = np.abs(target[i] - pred[i])
			sum_dif += dif
			n += 1
	mean_error = sum_dif / n
	print('Turn_error:', mean_error)
	if n >= 1:
		return mean_error
	else:
		return [-1]

def mdsave(model,model_path,index_num,sum_epoch,pos_range):
	model_name = str(index_num) + '_' + str(sum_epoch) + '_pos_' + str(pos_range) + '_lstm_model'  # 37_pos_40_lstm_model
	model.save(model_path + model_name + '.h5')
	model.save_weights(model_path + model_name + '_weights.h5')
def linear_predict(model_linear,pos_range,code = '601006'):
	pos_x, pos_target, pos_cls = getStockData.get_test_data(code=code, pos_range=pos_range)
	#pos_target = tpc.toTen(pos_target, 101)
	pos_x_cnn = np.reshape(pos_x, [pos_x.shape[0], 1, 100, 5])
	#pos_x_test,y_train = tpc.xTo2D(X_train,y_train,150)
	predict_ten = model_linear.predict([pos_x,pos_x_cnn])
	#believe_P = np.max(predict_ten,1) * 100
	predict_ten *= 100
	pos_target *= 100
	predict_ten = np.clip(predict_ten,0,100)
	pos_target = np.reshape(np.clip(pos_target,0,100),[len(pos_target),1])
	return predict_ten,pos_target,pos_cls

def fig_show(pos_range,predict_ten = None, pos_target = None,pos_cls = None, code = '601006'):
	if predict_ten is None or pos_target is None or pos_cls is None:
		predict_ten, pos_target,pos_cls = linear_predict(model_linear, pos_range, code=code)
	plt.plot(predict_ten,label='pred')
	plt.plot(pos_target,label='true')
	plt.plot(pos_cls * 3, label='close')
	#plt.plot(ave_pred,label='ave pred')
	plt.legend(loc='upper right')
	plt.rcParams['figure.figsize'] = (12.0, 7.0)
	plt.show()
	return plt
def fig_save(plt,code,model_path,index_num,sum_epoch,pos_range):
	plt.rcParams['savefig.dpi'] = 500
	model_name = str(index_num) + '_' + str(sum_epoch) + '_pos_' + str(pos_range) + '_lstm_model'
	plt.savefig(model_path + model_name + code + '.png', format='png',dpi=500)

def loadModel(model_path,index_n=1, epochs=2, pos_range=0.25):
	model_name = str(index_n) + '_' + str(epochs) + '_pos_' + str(pos_range) + '_lstm_model'  # 37_pos_40_lstm_model
	model = load_model(model_path + model_name + '.h5')
	model.load_weights(model_path + model_name + '_weights.h5')
	return model,model_name

def save_evaluate(model_linear,model_path,pos_range,test_codes,index,epoch,below = 5):
	eval_save = open(model_path + 'result.txt', 'a')
	eval_save.write('index: ' + str(index) + ' epoch:'+ str(epoch) + '\n')
	eval_result = [['code','tac','wmae','TE','Recall']]
	recall_l = []
	tac_l = []
	wmae_l = []
	te_l = []

	for code in test_codes:
		pred, pos_target,pos_cls = linear_predict(model_linear, pos_range, code=code)
		recall = Recall(pred, pos_target, below=below)
		tac = TAC(pred, pos_target)
		wmae = WMAE(pred, pos_target)
		te = Turn_error(pred, pos_target,below = below)
		temeval = {'code':code,'Turn Error':te[0],'recall:':recall[0],'tac':tac, 'wmae':wmae}
		eval_save.write(str(temeval) + '\n')
		eval_result.append([code,tac,wmae,te,recall])
		temeval = {}
		recall_l.append(recall[0])
		te_l.append(te[0])
		wmae_l.append(wmae)
		tac_l.append(tac)
	#eval_save.write('eval_result: ' + str(eval_result) + '\n')
	while -1 in recall_l:
		recall_l.remove(-1)
	while -1 in te_l:
		te_l.remove(-1)
	ave_recall = np.average(recall_l)
	ave_turn_error = np.average(te_l)
	eval_save.write('ave_recall:'+ str(ave_recall) + '  ave_turn error:' + str(ave_turn_error)+'\n')
	eval_save.close()
	return eval_result,[recall_l,te_l,wmae_l,tac_l,ave_recall,ave_turn_error]


if __name__=='__main__':
	global_start_time = time.time()
	epochs = 17
	seq_len = 100
	POS = 'pos'
	CLS = 'cls'
	model_path = 'model_linear/'
	train_mode = POS



	print('> Loading data... ')
	#nor_result = result
	path = 'datafiles/'

	pos_range=0.2
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

	#y_test = tpc.toTen(y_test,101)
	#y_train = tpc.toTen(y_train,101)

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

	#model_linear,model_name = loadModel(model_path,index_n=1, epochs=62, pos_range=0.2)
	soft_linear_model,linear_model = lstm.share_model_linear()
	model_linear = linear_model
	model_linear.summary()

	index_num = 1
	sum_epoch = 62  #234
	#loss_model = losses.distance_categorical_crossentropy
	#loss_model = losses.mae_categorical_crossentropy
	# loss_model = losses.categorical_crossentropy
	#loss_linear = losses.weight_mean_absolute_error
	loss_linear = losses.mae
	#rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
	opti = optimizers.Adam(lr=0.0001)
	#model.compile(loss=loss_model, optimizer=rmsprop, metrics=['accuracy'])
	model_linear.compile(loss=loss_linear, optimizer=opti, metrics=['accuracy'])
	for _ in range(2):
		#pre_loss = new_loss.copy()
		for _ in range(4):
			epochs =1
			hist = model_linear.fit(
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
			mdsave(model_linear,model_path, index_num, sum_epoch, pos_range)

			code = '601699'
			pred,pos_target,pos_cls = linear_predict(model_linear,pos_range, code=code)
			pltt = fig_show(pos_range,pred,pos_target,pos_cls,code=code)
			fig_save(pltt, code, model_path, index_num, sum_epoch, pos_range)
			pltt.close()
			evaluates, music = save_evaluate(model_linear, model_path, pos_range, sz50_code[11:], index=index_num,
											 epoch=sum_epoch, below=5)
		sz50 = pd.read_csv('sz50_code.csv')
		sz50_code = sz50['code50'].tolist()
		sz50_code = list(map(str,sz50_code))


		tn_error = np.array(music[1])
		recall_error = np.array(music[0])
		avge_tn_error = np.average(tn_error)
		avge_recall_error = np.average(recall_error)



		# model_name = '0_8_pos_0.4_lstm_model'  # 37_pos_40_lstm_model
		# model_name = '0_minErr_19_pos_0.15_lstm_model'
		# model = load_model('modelfiles2/'+ model_name + '.h5')
		# model.load_weights('modelfiles2/'+ model_name + '_weights.h5')

		#predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)

		#print('Training duration (s) : ', time.time() - global_start_time)
		#plot_results_multiple(predictions, y_test, 49)
		sum_eval=0
		eval_save = open(model_path + 'result.txt', 'a')
		code_list=['002154','000536','600108','601006','600887','600773']
		#code='601006'
		for code in code_list:
			# try:
			# 	pos_x, pos_target,pos_cls = getStockData.get_test_data(code=code,pos_range=pos_range)
			# except:
			# 	continue
			pos_x, pos_target, pos_cls = getStockData.get_test_data(code=code, pos_range=pos_range)
			#pos_target = tpc.toTen(pos_target, 101)
			pos_x_cnn = np.reshape(pos_x, [pos_x.shape[0], 1, 100, 5])
			#pos_x_test,y_train = tpc.xTo2D(X_train,y_train,150)

			predict_ten = model_linear.predict([pos_x,pos_x_cnn])
			#believe_P = np.max(predict_ten,1) * 100
			predict_ten *= 100
			pos_target *= 100
			plt.plot(predict_ten,label='pred')
			plt.plot(pos_target,label='true')
			plt.plot(pos_cls * 3, label='close')
			#plt.plot(ave_pred,label='ave pred')
			plt.legend(loc='upper right')
			plt.rcParams['figure.figsize'] = (12.0, 7.0)
			plt.show()
			plt.rcParams['savefig.dpi'] = 500
			model_name = str(index_num) + '_' + str(sum_epoch) + '_pos_' + str(pos_range) + '_lstm_model'
			plt.savefig(model_path + model_name + code + '.png', format='png',dpi=500)
			plt.close()
			eval_result,over_eval = evalute_result(predict_pos,y_pos)
			sum_eval += eval_result
			result_sum = 'epoch:'+str(sum_epoch)+' loss:'+str(new_loss)+' evaluate:'+str(eval_result)+'sumeva' +  str(over_eval) +'code:'+str(code) + '\n'
			eval_save.write(result_sum)
		ave_eval = sum_eval / len(code_list)
		eval_save.write('average evaluation: ' + str(ave_eval) + 'index: ' +str(index_num) + '\n')
		eval_save.close()


	pre_mean_bottom = evalute_result(predict_pos,y_pos)
	min_mean_bottom = pre_mean_bottom.copy()
	lrs = 0.0001

	for _ in range(5):
		epochs = 10
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












