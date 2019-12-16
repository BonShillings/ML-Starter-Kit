import numpy as np
from ml_kit.loaders.list_loader import ListLoader

class TabularFileLoader(ListLoader):

	def __init__(self, data_source, batch_size):

		data_file = data_source[0]
		label_file = data_source[1]

		with open(data_file) as r:
			rows = r.readlines()
			rows = [[row.split('\t')] for row in rows]

		with open(label_file) as l:
			labels = l.readlines()
			for i in range(len(labels)):
				rows[i].append(labels[i])

		super(TabularFileLoader, self).__init__(rows, batch_size)