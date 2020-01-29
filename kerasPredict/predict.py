import keras.models as KModel
import dataProcess.DataPreProcess as dpp
import numpy as np
import matplotlib.pyplot as plt
import kerasPredict.model_process as mp


def predict_top():
	# 加载之前训练好的模型
	model_path = '../model_linear/top_model/'
	index = 1
	sum_epoch = 12
	pos_range = 0.2
	model_linear, model_name = mp.loadModel(model_path, index_n=index, epochs=sum_epoch, pos_range=pos_range)

	stock_d = {'华东医药': '000963',
			   '中国巨石': '600176',
			   '中国石化': '600028',
			   '旗滨集团': '601636',
			   '宝钢股份': '600019',
			   '奥康国际': '603001'
			   }
	predict_name = '中国巨石'
	stock_name_l = stock_d.keys()
	for stock_name in stock_name_l:
		print('预测：' + stock_name)
		# 获取单只股票数据
		pos_x, pos_target, cls = dpp.get_test_data(code=stock_d[stock_name], seq_len=100, pos_range=0.2, ktype='D')
		xcnn_train = np.reshape(pos_x, [pos_x.shape[0], 1, 100, 5])

		# 计算预测结果
		predict_ten = model_linear.predict([pos_x, xcnn_train])
		predict_ten *= 100
		pos_target = pos_target * 100
		predict_ten = np.clip(predict_ten, 0, 100)
		pos_target = np.reshape(np.clip(pos_target, 0, 100), [len(pos_target), 1])

		# 绘图
		plt.plot(predict_ten, label='pred')
		plt.plot(pos_target, label='true')
		plt.plot(cls, label='close')
		plt.plot([10] * len(cls))
		plt.plot([5] * len(cls))

		plt.legend(loc='upper right')
		plt.rcParams['figure.figsize'] = (12.0, 7.0)
		# plt.show()

		figer_path = '../figer_result/stock/top/'
		fig_name = str(index) + '_' + str(sum_epoch) + '_pos_' + str(pos_range) + '_' + stock_name
		plt.savefig(figer_path + fig_name + '.png', format='png', dpi=300)
		# plt.show()
		plt.close()

def predict_bottom():
	#加载之前训练好的模型
	model_path = '../model_linear/'
	index = 1
	sum_epoch = 125
	pos_range = 0.2
	model_linear, model_name = mp.loadModel(model_path, index_n = index, epochs = sum_epoch, pos_range = pos_range)

	stock_d = {'华东医药':'000963',
			   '中国巨石':'600176',
			   '中国石化':'600028',
			   '旗滨集团':'601636',
			   '宝钢股份':'600019',
			   '奥康国际': '603001'
			   }
	predict_name = '中国巨石'
	stock_name_l = stock_d.keys()
	for stock_name in stock_name_l:
		print('预测：'+ stock_name)
		#获取单只股票数据
		pos_x,pos_target,cls = dpp.get_test_data(code = stock_d[stock_name],seq_len=100, pos_range=0.2, ktype='D')
		xcnn_train = np.reshape(pos_x, [pos_x.shape[0], 1, 100, 5])

		#计算预测结果
		predict_ten = model_linear.predict([pos_x, xcnn_train])
		predict_ten *= 100
		pos_target = pos_target * 100
		predict_ten = np.clip(predict_ten, 0, 100)
		pos_target = np.reshape(np.clip(pos_target, 0, 100), [len(pos_target), 1])

		#绘图
		plt.plot(predict_ten,label='pred')
		plt.plot(pos_target,label='true')
		plt.plot(cls, label='close')
		plt.plot([10]*len(cls))
		plt.plot([5] * len(cls))

		plt.legend(loc='upper right')
		plt.rcParams['figure.figsize'] = (12.0, 7.0)
		#plt.show()

		figer_path = '../figer_result/stock/'
		fig_name = str(index) + '_' + str(sum_epoch) + '_pos_' + str(pos_range) + '_' + stock_name
		plt.savefig(figer_path + fig_name + '.png', format='png', dpi=300)
		# plt.show()
		plt.close()

def predict_all():
	#加载之前训练好的模型
	model_path_top = '../model_linear/top_model/'
	index_top = 1
	sum_epoch_top = 12
	pos_range_top = 0.2
	model_top, model_name_top = mp.loadModel(model_path_top, index_n=index_top, epochs=sum_epoch_top, pos_range=pos_range_top)

	model_path = '../model_linear/'
	index = 1
	sum_epoch = 125
	pos_range = 0.2
	model_bottom, model_name = mp.loadModel(model_path, index_n = index, epochs = sum_epoch, pos_range = pos_range)

	stock_d = {'华东医药':'000963',
			   '中国巨石':'600176',
			   '中国石化':'600028',
			   '旗滨集团':'601636',
			   '宝钢股份':'600019',
			   '奥康国际': '603001'
			   }
	predict_name = '中国巨石'
	stock_name_l = stock_d.keys()
	for stock_name in stock_name_l:
		print('预测：'+ stock_name)
		#获取单只股票数据
		pos_x,pos_target,cls = dpp.get_test_data(code = stock_d[stock_name],seq_len=100, pos_range=0.2, ktype='D')
		xcnn_train = np.reshape(pos_x, [pos_x.shape[0], 1, 100, 5])

		#计算预测结果
		predict_top_result = model_top.predict([pos_x, xcnn_train])  #计算顶部预测结果
		predict_bottom_result = model_bottom.predict([pos_x, xcnn_train])#计算底部预测结果

		predict_top_result *= 100
		predict_bottom_result *= 100

		predict_top_result = np.clip(predict_top_result, 0, 100)
		predict_bottom_result = np.clip(predict_bottom_result, 0, 100)

		#处理原始目标值
		pos_target = pos_target * 100
		pos_target = np.reshape(np.clip(pos_target, 0, 100), [len(pos_target), 1])

		#绘图
		plt.plot(predict_top_result, label='pred_top')
		plt.plot(predict_bottom_result, label='pred_bottom')

		plt.plot(pos_target, label='true')
		plt.plot(cls, label='close')
		plt.plot([10]*len(cls))
		plt.plot([5] * len(cls))
		plt.plot([60] * len(cls))

		plt.legend(loc='upper right')
		plt.rcParams['figure.figsize'] = (12.0, 7.0)
		#plt.show()

		figer_path = '../figer_result/stock/'
		fig_name = str(index) + '_' + str(sum_epoch) + '_pos_' + str(pos_range) + '_' + stock_name
		plt.savefig(figer_path + fig_name + '.png', format='png', dpi=300)
		# plt.show()
		plt.close()


if __name__ == '__main__':
	predict_all()




