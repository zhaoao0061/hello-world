import dataProcess.DataPreProcess as getData


class RunPredict:
	def __init__(self,epochs):
		self.epochs = epochs

	def run(self):
		self.seq_len = 100  # 设置序列长度
		self.features = 3
		self.path = 'data/'  # 设置数据存储路径TabError: inconsistent use of tabs and spaces in indentation

		stockCode = getData.getStockCode()  # 获取训练集股票代码，具体哪个代码去函数里设置
		stockCode = stockCode  # 股票编号

		if True:  # 保存或加载数据，True为在线获取数据，处理后保存于path目录下 。False为直接加载之前存储的数据。
			file_name = self.path + 'pos_'  # pos_40_train_z.npz
			result, train = getData.data_save(stockCode, seqlen = self.seq_len, file_name = file_name)
		else:  # 加载文件
			file_name = 'pos_'
			train = np.load(path + file_name + 'train_len' + str(seq_len) + '.npz')

		X_train, y_train = getData.map_to_train(train)

		print("X_train:", X_train)
		print("X_trian的长度：", len(X_train))
		print("y_train:", y_train)
		print("y_train的长度：", len(y_train))

		loss_linear = losses.mse
		opti = optimizers.Adam(lr=0.01)  # 设置学习率，通常从0.001开始，逐步减小。或根据实际情况来，训练坏掉，就减小试试。
		model_linear.compile(loss=loss_linear, optimizer=opti, metrics=['accuracy'])  # 模型编译一下，就可以训练了。

		lossss = []

		optimizer = optim.Adam(Linear.parameters(), lr=0.001)
		loss_func = nn.MSELoss()

		for i in range(epoch):
			# m = RNN()
			running_loss = 0.0
			# for i in X_train.size():

			position = 0
			x_size = X_train.shape[0]
			while (position < x_size):

				optimizer.zero_grad()
				end = position + batch_size

				if end > x_size:
					end = x_size

				pred = Linear(X_train[position:end, :, :])  # torch.Size([1000, 1, 1])
				# loss = loss_func(out, targ_trajs)

				loss = loss_func(pred[:, :1, :], y_train[position:end])
				loss.backward()
				optimizer.step()
				running_loss += loss.item()
				position = end
			# print(loss.item())
			# print(running_loss)

			print('Epoch:{}, Loss:{:.5f}'.format(i + 1, running_loss))

			pred = Linear(X_train[:300, :, :])
			x = pred[:300, :1, :].cpu().detach().numpy()  # 将格式转化为numpy
			y = y_train[:300, :1, :].cpu().detach().numpy()  # 将格式化转化为numpy

			# X_num = x.size()
			x = x.reshape(300)
			y = y.reshape(300)

			plt.plot(x, label='true trajectory')
			plt.plot(y, label='learned trajectory (t>0)')

			plt.legend(loc='upper right')  # 绘制图例

			plt.savefig('./vis-nsample1-Linear.png', dpi=500)
			plt.close()


if __name__ == "__main__":
	run = RunPredict(5)