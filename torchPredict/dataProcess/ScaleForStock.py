


def normalise_windows(window_data):
	normalised_data = []
	for window in window_data:
		try:
			normalised_window = [((float(p) - float(window[0]))/ float(window[0]))/2 for p in window]
			normalised_data.append(normalised_window)
		except:
			print('ca')
	return  normalised_data

def dynamic_normalise_close(window_data):
	normalised_data = []
	normalised_cls = []
	for window in window_data:
		try:
			normalised_window = [(((float(p) - float(window[0]))/ float(window[0]))) for p in window]
			normalised_data.append(normalised_window)
		except:
			print('ca')
	max_ncls = max(normalised_data)
	min_ncls = min(normalised_data)
	bias = (max_ncls + min_ncls) / 2
	for cls in normalised_data:
		normalised_cls.append(cls - bias)
	return  normalised_cls


def _nor_volume(volume):#标准化成交量
	nor_vol=[]
	for i in range(len(volume)):
		min_vol = 0
		if i < 50:
			max_vol = max(volume[:50])
			nor_vol.append(volume[i] / (max_vol - min_vol))
		else:
			max_vol = max(volume[:i])
			nor_vol.append(volume[i] / (max_vol - min_vol))
	return nor_vol