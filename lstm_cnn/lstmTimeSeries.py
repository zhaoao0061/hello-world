import os
import time
import warnings
import numpy as np
import keras
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import optimizers,losses
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, Flatten,Permute,Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Embedding
from keras.models import Model
from keras.layers.merge import concatenate
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def load_data(filename, seq_len, normalise_window):
	f = open(filename, 'rb').read()
	data = f.decode().split('\n')
	sequence_length = seq_len + 1
	result = []
	for index in range(len(data) - sequence_length):
		result.append(data[index:index+sequence_length])

	if normalise_window:
		result = normalise_windows(result)

	result = np.array(result)

	row = round(0.9 * result.shape[0])
	train = result[:int(row), :]
	np.random.shuffle(train)
	x_train = train[:, :-1]
	y_train = train[:, -1]
	x_test = result[int(row):, :-1]
	y_test = result[int(row):, -1]

	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

	return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
	normalised_data = []
	for window in window_data:
		normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
		normalised_data.append(normalised_window)
	return  normalised_data

def build_model(nums,layers,loss_model=losses.pos_error):
	model = Sequential()

	model.add(LSTM(
		input_shape = (layers[1], layers[0]),
		output_dim = layers[1],
		return_sequences=True,
		activation='relu',
	))
	model.add(Dropout(0.5))

	for i in range(2,nums-2):
		model.add(LSTM(
			layers[i],
			return_sequences=True,
			activation='relu',
		))
		model.add(Dropout(0.5))

	model.add(LSTM(
		layers[-2],
		return_sequences=False,
		activation='relu',
	))
	model.add(Dropout(0.5))

	model.add(Dense(output_dim = layers[-1]))
	model.add(Activation("linear"))

	model.add(Dense(101))
	model.add(Activation('softmax'))

	model.summary()

	start = time.time()
	#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9,nesterov=True)
	rmsprop = optimizers.RMSprop(lr=0.001,rho=0.9,epsilon=1e-06)
	# loss_model = losses.pos_error
	# loss_model_mae = losses.mean_absolute_error
	model.compile(loss = loss_model, optimizer = rmsprop, metrics=['accuracy'])
	#model.compile(loss = 'mse', optimizer=rmsprop)
	print("> Compilation Time : ", time.time() - start)
	return  model


def cnn_model(x_train,class_num=101):
	model = Sequential()

	model.add(Conv2D(5, (4, 3), padding='valid',
					 data_format='channels_first',
					 input_shape=x_train.shape[1:]))  # filters=32, kernel_size=(3,3),input_shape = (32,32,3)
	model.add(Activation('relu'))

	# model.add(MaxPooling2D(pool_size=(2, 1),
	# 				 data_format='channels_first'))
	# model.add(Dropout(0.25))

	model.add(Conv2D(10, (2, 2), padding='valid',
					 data_format='channels_first'))
	model.add(Activation('relu'))

	model.add(Conv2D(20, (2, 2), padding='valid',
					 data_format='channels_first'))
	model.add(Activation('relu'))

	# model.add(MaxPooling2D(pool_size=(2, 1),
	# 				 data_format='channels_first'))
	# model.add(Dropout(0.25))

	# model.add(Conv2D(60, (3, 3), padding='same',
	# 				 data_format='channels_first'))
	# model.add(Activation('relu'))
	#
	# model.add(MaxPooling2D(pool_size=(2, 2),
	# 				 data_format='channels_first'))
	# model.add(Dropout(0.25))

	model.add(Flatten(data_format='channels_first'))

	model.add(Dense(100))
	model.add(Activation('linear'))
	model.add(Dropout(0.35))

	model.add(Dense(101))
	model.add(Activation('softmax'))

	model.summary()

	return model

def cnn_model2(x_train,class_num=101):
	cnn_input = Input(shape=[1,100,5], name='cnn_input')
	conv1 = Conv2D(5, (4, 3),
				   # padding='same',
				   data_format='channels_first')(cnn_input)
	conv2 = Conv2D(10, (2, 2),
				   # padding='same',
				   data_format='channels_first')(conv1)
	conv3 = Conv2D(20, (2, 2),
				   # padding='same',
				   data_format='channels_first')(conv2)
	# pool = MaxPooling2D(pool_size=(3,1),data_format='channels_first')(conv1)
	reshape = Reshape((20, 95))(conv3)
	permt = Permute((2, 1))(reshape)
	cnn_lstm = LSTM(
		150,
		return_sequences=False,
		activation='relu', )(permt)  # 100

	# flat1 = Flatten(data_format='channels_first')(conv3)
	cnn_dense1 = Dense(100, activation='relu', name='cnn_dense1')(cnn_lstm)
	cnn_dense2 = Dense(50, activation='relu', name='cnn_dense2')(cnn_dense1)
	cnn_dense3 = Dense(50, activation='linear', name='cnn_dense3')(cnn_dense2)

	output = Dense(101, activation='softmax')(cnn_dense3)
	model = Model(inputs=cnn_input, outputs=output)

	return model



def share_model(class_num = 101):
###################################
#lstm layer
	layers = [5,
			  100,  #input(5,100)
			  50,   #lstm1(100,50)
			  100,  #lstm2(100,100)
			  150,  #lstm3(150)
			  150,
			  50,
			  50,
			  101]
	lstm_input = Input(shape=(layers[1], layers[0]),name='lstm_input')#(100,5)

	lstm1 = LSTM(
			layers[2],
			return_sequences=True,
			activation='relu',)(lstm_input)   #(100,50)
	lstm2 = LSTM(
			layers[3],
			return_sequences=True,
			activation='relu',)(lstm1)    #(100,100)
	lstm_end = LSTM(
			layers[4],
			return_sequences=False,
			activation='relu', )(lstm2)   #(150,)

####################################
#cnn layer
	cnn_layers = [5,
				  10,
				  20,
				  layers[4]]
	cnn_input = Input(shape=[1,layers[1], layers[0]],name='cnn_input')#(none,1,100,5)
	conv1 = Conv2D(cnn_layers[0], (4, 3),
			#padding='same',
		    data_format='channels_first')(cnn_input)#(none,10,97,3)
	conv2 = Conv2D(cnn_layers[1], (2, 2),
			#padding='same',
		    data_format='channels_first')(conv1)   #(none,20,96,2)
	conv3 = Conv2D(cnn_layers[2], (2, 2),
			#padding='same',
		    data_format='channels_first')(conv2)   #(none,50,95,1)
	#pool = MaxPooling2D(pool_size=(3,1),data_format='channels_first')(conv1)
	reshape = Reshape((cnn_layers[2],95))(conv3)   #(none,50,95)
	permt = Permute((2, 1))(reshape)    #(none,95,50)
	cnn_lstm1 = LSTM(
		100,
		return_sequences=True,
		activation='relu', )(permt)  # (95,100)
	cnn_lstm2 = LSTM(
		100,
		return_sequences=True,
		activation='relu', )(cnn_lstm1)  # (95,100)
	cnn_lstm_out = LSTM(
			cnn_layers[-1],
			return_sequences=False,
			activation='relu',)(permt) #(150)
	#flat1 = Flatten(data_format='channels_first')(conv3)
	# cnn_dense = Dense(layers[-2],activation='linear',name='cnn_dense')(cnn_lstm)


####################################
#merge layer
	merge_layers=[layers[4],
				  100,
				  50,
				  50]
	merge = concatenate([lstm_end,cnn_lstm_out])  #150+150 = (300,)

	#4个Dense全连接层
	hidden1 = Dense(100,activation='relu')(merge)#100 relu
	hidden2 = Dense(50,activation='relu')(hidden1)   #50
	hidden_end = Dense(50, activation='linear')(hidden2)#50 linear

	output = Dense(class_num,activation='softmax')(hidden_end)


####################################
	model = Model(inputs=[lstm_input,cnn_input],outputs = output)
	model.summary()
	#keras.utils.plot_model(model, to_file='model/modeltest4.png')


	output2 = Dense(layers[-1],activation='linear')(output)
	model2 = Model(inputs=[lstm_input, cnn_input], outputs=output2)

	return model,model2
#--------------------------------------------------------------------------------------#





def share_model3(class_num = 101):
###################################
#lstm layer
	layers = [5,100,  #input(5,100)
			  100,   #lstm1(100,50)
			  100,  #lstm2(100,100)
			  150]
	lstm_input = Input(shape=(layers[1], layers[0]),name='lstm_input')#(100,5)

	lstm1 = LSTM(
			100,
			return_sequences=True,
			activation='relu',)(lstm_input)   #(100,100)
	lstm2 = LSTM(
			100,
			return_sequences=True,
			)(lstm1)    #(100,100)
	lstm_end = LSTM(
			100,
			return_sequences=False,
			activation='relu', )(lstm2)   #(150,)

####################################
#cnn layer
	cnn_layers = [5,
				  10,
				  20,
				  layers[4]]
	cnn_input = Input(shape=[1,layers[1], layers[0]],name='cnn_input')#(none,1,100,5)
	conv1 = Conv2D(cnn_layers[0], (4, 3),
			#padding='same',
		    data_format='channels_first')(cnn_input)#(none,10,97,3)
	conv2 = Conv2D(cnn_layers[1], (2, 2),
			#padding='same',
		    data_format='channels_first')(conv1)   #(none,20,96,2)
	conv3 = Conv2D(cnn_layers[2], (2, 2),
			#padding='same',
		    data_format='channels_first')(conv2)   #(none,50,95,1)
	#pool = MaxPooling2D(pool_size=(3,1),data_format='channels_first')(conv1)
	reshape = Reshape((cnn_layers[2],95))(conv3)   #(none,50,95)
	permt = Permute((2, 1))(reshape)    #(none,95,50)
	# cnn_lstm1 = LSTM(
	# 	100,
	# 	return_sequences=True,
	# 	activation='relu', )(permt)  # (95,100)
	# cnn_lstm2 = LSTM(
	# 	100,
	# 	return_sequences=True,
	# 	activation='relu', )(cnn_lstm1)  # (95,100)
	cnn_lstm_out = LSTM(
			cnn_layers[-1],
			return_sequences=False,
			activation='relu',)(permt) #(150,)


####################################
#merge layer

	merge = concatenate([lstm_end,cnn_lstm_out])  #150+150 = (300,)

	#4个Dense全连接层
	hidden1 = Dense(200,activation='relu')(merge)#100 relu
	dp1 = Dropout(0.3)(hidden1)
	hidden2 = Dense(100,activation='relu')(dp1)   #50
	dp2 = Dropout(0.3)(hidden2)
	hidden_end = Dense(100, activation='linear')(hidden2)#50 linear

	output = Dense(class_num,activation='softmax')(hidden_end)


####################################
	model = Model(inputs=[lstm_input,cnn_input],outputs = output)
	model.summary()
	keras.utils.plot_model(model, to_file='modeltest4.png')

####################################
	output2 = Dense(50,activation='linear')(output)
	model2 = Model(inputs=[lstm_input, cnn_input], outputs=output2)

	return model,model2

def share_model_linear(class_num = 101):
###################################
#lstm
	layers = [5,100,  #input(5,100)
			  100,   #lstm1(100,50)
			  100,  #lstm2(100,100)
			  150]
	lstm_input = Input(shape=(layers[1], layers[0]),name='lstm_input')#(100,5)

	lstm1 = LSTM(
			100,
			return_sequences=True,
			activation='elu',)(lstm_input)   #(100,100)relu
	lstm2 = LSTM(
			100,
			return_sequences=True,
			activation='elu',)(lstm1)    #(100,100)
	lstm_end = LSTM(
			150,
			return_sequences=False,
			activation='elu', )(lstm2)   #(150,)relu

####################################
#cnn layer
	cnn_layers = [5,
				  10,
				  20,
				  layers[4]]
	cnn_input = Input(shape=[1,layers[1], layers[0]],name='cnn_input')#(none,1,100,5)
	conv1 = Conv2D(cnn_layers[0], (4, 3),
			#padding='same',
		    data_format='channels_first')(cnn_input)#(none,10,97,3)
	conv2 = Conv2D(cnn_layers[1], (2, 2),
			#padding='same',
		    data_format='channels_first')(conv1)   #(none,20,96,2)
	conv3 = Conv2D(cnn_layers[2], (2, 2),
			#padding='same',
		    data_format='channels_first')(conv2)   #(none,50,95,1)
	#pool = MaxPooling2D(pool_size=(3,1),data_format='channels_first')(conv1)
	reshape = Reshape((cnn_layers[2],95))(conv3)   #(none,50,95)
	permt = Permute((2, 1))(reshape)    #(none,95,50)
	# cnn_lstm1 = LSTM(
	# 	100,
	# 	return_sequences=True,
	# 	activation='relu', )(permt)  # (95,100)
	# cnn_lstm2 = LSTM(
	# 	100,
	# 	return_sequences=True,
	# 	activation='relu', )(cnn_lstm1)  # (95,100)
	cnn_lstm_out = LSTM(
			cnn_layers[-1],
			return_sequences=False,
			activation='elu',)(permt) #(150,)


####################################
#merge layer

	merge = concatenate([lstm_end,cnn_lstm_out])  #150+150 = (300,)

	#4个Dense全连接层
	hidden1 = Dense(200,activation='elu')(merge)#100 relu
	dp1 = Dropout(0.3)(hidden1)
	hidden2 = Dense(100,activation='elu')(dp1)   #50
	dp2 = Dropout(0.3)(hidden2)
	hidden_end = Dense(100, activation='tanh')(dp1)#50 linear
	linear_end = Dense(1,activation='tanh')(hidden_end)
	output1 = Dense(101,activation='softmax')(hidden_end)
	soft_linear_out = Dense(1, activation='linear')(output1)

####################################
	soft_linear_model = Model(inputs=[lstm_input,cnn_input],outputs = soft_linear_out)
	linear_model = Model(inputs=[lstm_input,cnn_input],outputs = linear_end)
	#keras.utils.plot_model(model, to_file='model/modeltest4.png')
	return soft_linear_model,linear_model


def share_model2(class_num = 101):
	layers = [5,
			  100,  #input(5,100)
			  20,   #lstm1(100,20)
			  50,  #lstm2(100,50)
			  150,  #lstm3(150)
			  150,
			  50,
			  50,
			  101]
	lstm_input = Input(shape=(layers[1], layers[0]),name='lstm_input')#(100,5)

	lstm1 = LSTM(
			layers[2],
			return_sequences=True,
			activation='relu',)(lstm_input)   #100
	lstm2 = LSTM(
			layers[3],
			return_sequences=True,
			activation='relu',)(lstm1)    #100
	lstm_end = LSTM(
			layers[4],
			return_sequences=False,
			activation='relu', )(lstm2)   #150

####################################
	cnn_layers = [5,
				  10,
				  20,
				  layers[4]]
	cnn_input = Input(shape=[1,layers[1], layers[0]],name='cnn_input')#(none,1,100,5)
	conv1 = Conv2D(cnn_layers[0], (4, 3),
			padding='same',
		    data_format='channels_first')(cnn_input)  #(20,100,5)
	conv2 = Conv2D(cnn_layers[1], (2, 2),
			padding='same',
		    data_format='channels_first')(conv1)  #(20,100,5)
	conv3 = Conv2D(cnn_layers[2], (2, 2),
			padding='same',
		    data_format='channels_first')(conv2)  #(50,100,5)

	conv4 = Conv2D(cnn_layers[2], (1, 3),
				   #padding='same',
				   data_format='channels_first')(conv3)  # (50,100,3)
	conv5 = Conv2D(cnn_layers[2], (1, 2),
				   #padding='same',
				   data_format='channels_first')(conv4)  # (50,100,2)
	conv6 = Conv2D(cnn_layers[2], (1, 2),
				   #padding='same',
				   data_format='channels_first')(conv5)  # (50,100,1)
	#pool = MaxPooling2D(pool_size=(3,1),data_format='channels_first')(conv1)
	reshape = Reshape((cnn_layers[2],100))(conv3)
	permt = Permute((2, 1))(reshape)
	cnn_lstm = LSTM(
			cnn_layers[-1],
			return_sequences=False,
			activation='relu',)(permt) #150
	#flat1 = Flatten(data_format='channels_first')(conv3)
	# cnn_dense = Dense(layers[-2],activation='linear',name='cnn_dense')(cnn_lstm)


####################################
	merge_layers=[layers[4],
				  100,
				  50,
				  50]
	merge = concatenate([lstm_end,cnn_lstm])
	reshape2 = Reshape((2,layers[4]))(merge)
	permt2 = Permute((2, 1))(reshape2)

	hidden_lstm = LSTM(
			merge_layers[0],
			return_sequences=False,
			activation='relu', )(permt2)    #150

	#三个Dense全连接层
	hidden1 = Dense(merge_layers[1],activation='relu')(hidden_lstm)#100 relu
	hidden2 = Dense(merge_layers[2],activation='relu')(hidden1)   #50
	hidden_end = Dense(merge_layers[3], activation='linear')(hidden2)#50 linear

	output = Dense(class_num,activation='softmax')(hidden_end)
####################################
	model = Model(inputs=[lstm_input,cnn_input],outputs = output)
	model.summary()
	#keras.utils.plot_model(model, to_file='model/modeltest4.png')

####################################
	output2 = Dense(layers[-1],activation='linear')(hidden2)
	model2 = Model(inputs=[lstm_input, cnn_input], outputs=output2)

	return model,model2


def pos_error(y_true,y_pred):

    bias = 0.1
    time = 100
    middle = [0.8,0.2]
    c = K.round(y_pred)
    sig_up_p = 1 / ( 1 + K.exp(-(y_pred - middle[0]) * 24)) + bias
    sig_down_p = 1 / ( 1 + K.exp(-(y_pred - middle[1]) * 24)) - 1 + bias
    pred_val = c * sig_up_p + (1 - c) * sig_down_p

    c_t = K.round(y_true)
    sig_up_t = 1 / (1 + K.exp(-(y_true - middle[0]) * 24)) + bias
    sig_down_t = 1 / (1 + K.exp(-(y_true - middle[1]) * 24)) - 1 + bias
    true_val = c_t * sig_up_t + (1 - c_t) * sig_down_t

    return K.abs(pred_val - true_val) * time

def twodays_distants(y_true,y_pred):
    try:
        n1 = 5
        n2 = 4
        c = 0.1
        col = K.int_shape(y_pred)
        if col[1] > 1:
            next_div = y_true[0:, 1] - y_pred[0:, 0]
            next_div = K.sqrt(K.square(next_div + c**2))#求预测值与下一真实值的直线距离， 参数c 控制横向距离。
            abs_div = K.abs(y_true - y_pred)#
            abs_div0 = abs_div[0:, 0]
            first_loss = K.pow((n1 * abs_div0 + next_div),n2)
            print('Two days losses')
        else:
            first_loss = K.mean(K.square(y_pred - y_true), axis=-1)
            print('One day losses')
    except:
        print('some error occured in losses.py')
    else:
        return first_loss

def predict_sequences_multiple(model, data, window_size, prediction_len):
	#Predict sequence of 50 steps before shifting prediction run forward by 50 steps
	prediction_seqs = []
	for i in range(int(len(data)/prediction_len)):
		curr_frame = data[i*prediction_len]
		predicted = []
		for j in range(prediction_len):
			predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
		prediction_seqs.append(predicted)
	return prediction_seqs

def predict_point_by_point(model, data):
	predicted = model.predict(data)
	print('predicted shape:',np.array(predicted).shape) #(412L,1L)
	predicted = np.reshape(predicted, (predicted.size,))
	return predicted




def distance_categorical_crossentropy(target, output, from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.

    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output, #按照行的维度求和
                                len(output.get_shape()) - 1,
                                True)  #将每行中每个元素映射到  0-1  上, 使得每行总和为 1
        # manual computation of crossentropy
        _epsilon = tf.convert_to_tensor(epsilon(), output.dtype.base_dtype)  # epsilion() : return  _EPSILON = 1e-7，然后将其转换为tensor
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon) # tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。
        pos_true = tf.argmax(target,1)  #获取每行最大值的位置索引,即最终预测值
        pos_pred = tf.argmax(output,1)
        distance = tf.abs(pos_true - pos_pred)
        pos_true = pos_true/50 - 1
        pos_pred = pos_pred/50 - 1
        new_pos_true = 0.5 * (tf.log((1 + pos_true+ 0.01)/(1-pos_true + 0.01)))# artanhx函数  范围：[-2.62165,2.62165]
        new_pos_pred = 0.5 * (tf.log((1 + pos_pred+ 0.01)/(1-pos_pred + 0.01)))
        new_distance = tf.abs(new_pos_true - new_pos_pred)


        return - tf.reduce_sum((target * tf.log(output) + (1 - target) * tf.log(1-output)) * (new_distance + 1),
                               len(output.get_shape()) - 1)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)










