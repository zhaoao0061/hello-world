import kerasPredict.model.lstmTimeSeries as lstm
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras import optimizers,losses
import kerasPredict.Evaluate as ev

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu':0})))


def mdsave(model,model_path,index_num,sum_epoch,pos_range):
   model_name = str(index_num) + '_' + str(sum_epoch) + '_pos_' + str(pos_range) + '_lstm_model'  # 37_pos_40_lstm_model
   model.save(model_path + model_name + '.h5')
   model.save_weights(model_path + model_name + '_weights.h5')


def fig_show(predict_ten, pos_target, pos_cls = None):
   plt.plot(predict_ten,label='pred')
   plt.plot(pos_target,label='true')
   # plt.plot(pos_cls * 3, label='close')
   #plt.plot(ave_pred,label='ave pred')
   plt.legend(loc='upper right')
   plt.rcParams['figure.figsize'] = (12.0, 7.0)
   plt.show()
   return plt

def fig_save(plt,model_path,index_num,sum_epoch,pos_range):
   plt.rcParams['savefig.dpi'] = 500
   model_name = str(index_num) + '_' + str(sum_epoch) + '_pos_' + str(pos_range) + '_lstm_model'
   plt.savefig(model_path + model_name + '.png', format='png',dpi=500)

def loadModel(model_path,index_n=1, epochs=2, pos_range=0.25):
   model_name = str(index_n) + '_' + str(epochs) + '_pos_' + str(pos_range) + '_lstm_model'  # 37_pos_40_lstm_model
   model = load_model(model_path + model_name + '.h5')
   model.load_weights(model_path + model_name + '_weights.h5')
   return model,model_name



import dataProcess.FeaturesGen as FG

if __name__=='__main__':
   global_start_time = time.time()
   epochs = 17
   seq_len = 100

   model_path = '../model_linear/top_model/'
   POS = 'pos'
   CLS = 'cls'
   train_mode = POS


   print('> Loading data... ')
   #nor_result = result
   path = '../dataProcess/data_file/'

   pos_range = 0.2
   is_save = 0
   if is_save == 1:
      (X_train, y_train, X_test, y_test) = FG.get_train_save(path, pos_range) #从互联网下载数据并预处理
   else:
      file_name = '../dataProcess/data_file/pos_0.2_train_pos_z.npz'
      (X_train, y_train, X_test, y_test) = FG.get_train_load(file_name = file_name)
   print('> Data Loaded. Compiling...')

   i = 0
   for y in y_train:
      if y < 0.5:
         y_train[i] = 0.5
      # elif y <= 0.8:
      #    y_train[i] = 0.8
      # elif y <= 0.9:
      #    y_train[i] = 0.9
      i += 1

   xcnn_train = np.reshape(X_train, [X_train.shape[0], 1, 100, 5])
   xcnn_test = np.reshape(X_test, [X_test.shape[0], 1, 100, 5])

   start = time.time()
   input_nodes = X_train.shape[2]
   if y_train.ndim == 1:
      output_nodes = 1
   else:
      output_nodes = y_train.shape[1]


   index_num = 1
   sum_epoch = 0  #234

   #从之前训练的模型中加载
   # model_linear,model_name = loadModel(model_path,index_n=1, epochs = sum_epoch, pos_range=0.2)
   model_linear = lstm.share_model_linear()  #创建新模型
   model_linear.summary()


   #loss_model = losses.distance_categorical_crossentropy
   #loss_model = losses.mae_categorical_crossentropy
   #loss_model = losses.categorical_crossentropy
   #loss_linear = losses.weight_mean_absolute_error
   loss_linear = losses.mae
   #rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
   opti = optimizers.Adam(lr=0.0005)
   #model.compile(loss=loss_model, optimizer=rmsprop, metrics=['accuracy'])
   model_linear.compile(loss=loss_linear, optimizer=opti, metrics=['accuracy'])
   for _ in range(2):
      #pre_loss = new_loss.copy()
      for _ in range(2):
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
         mdsave(model_linear, model_path, index_num, sum_epoch, pos_range)

         predict_ten = model_linear.predict([X_test, xcnn_test])
         predict_ten *= 100
         pos_target = y_test * 100
         predict_ten = np.clip(predict_ten, 0, 100)
         pos_target = np.reshape(np.clip(pos_target, 0, 100), [len(pos_target), 1])

         plt.plot(predict_ten,label='pred')
         plt.plot(pos_target,label='true')

         plt.legend(loc='upper right')
         plt.rcParams['figure.figsize'] = (12.0, 7.0)
         # plt.show()

         plt.rcParams['savefig.dpi'] = 300
         model_name = str(index_num) + '_' + str(sum_epoch) + '_pos_' + str(pos_range) + '_lstm_model'
         figer_path = '../figer_result/top_pre_fig/top_'
         plt.savefig(figer_path + model_name + '.png', format='png', dpi=300)
         plt.close()

         # ##评估  计算查准率，下穿18
         # turn_error_value = ev.Turn_error(predict_ten, pos_target,8)
         # ev_save_text = '下穿8 turn_error: ' + str(turn_error_value)
		 #
         # up_turn_error_value = ev.up_Turn_error(predict_ten, pos_target,10)
         # ev_save_text = ev_save_text + '\n 上穿10 turn_error: ' + str(up_turn_error_value)
		 #
         # ev.save_result(ev_save_text, model_path = figer_path, index = index_num, epoch = sum_epoch)




