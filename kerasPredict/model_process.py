import keras.models as KModels

def mdsave(model,model_path,index_num,sum_epoch,pos_range):
	# 文件名格式： 37_pos_40_lstm_model
	model_name = str(index_num) + '_' + str(sum_epoch) + '_pos_' + str(pos_range) + '_lstm_model'  # 37_pos_40_lstm_model
	model.save(model_path + model_name + '.h5')
	model.save_weights(model_path + model_name + '_weights.h5')


def loadModel(model_path, index_n, epochs, pos_range):
	# 文件名格式： 37_pos_40_lstm_model
	model_name = str(index_n) + '_' + str(epochs) + '_pos_' + str(pos_range) + '_lstm_model'  #文件名格式： 37_pos_40_lstm_model
	model = KModels.load_model(model_path + model_name + '.h5')
	model.load_weights(model_path + model_name + '_weights.h5')
	return model,model_name