import matplotlib.pyplot as plt


def fig_save(plt, model_path, index_num, sum_epoch, pos_range):
	plt.rcParams['savefig.dpi'] = 500
	model_name = str(index_num) + '_' + str(sum_epoch) + '_pos_' + str(pos_range) + '_lstm_model'
	plt.savefig(model_path + model_name + '.png', format='png',dpi=500)