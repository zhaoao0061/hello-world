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
####################################
	linear_model = Model(inputs=[lstm_input,cnn_input],outputs = linear_end)
	keras.utils.plot_model(linear_model, to_file='model/modeltest4.png')
	return linear_model


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










