import numpy as np
from loaders.abstract_loader import AbstractLoader

class ListLoader(AbstractLoader):

	def __init__(self, data_source, batch_size):
		super(ListLoader, self).__init__(data_source, batch_size)

		self.X = np.array([row[0] for row in data_source])
		self.y = np.array([row[1] for row in data_source])
		self.pointer = 0

	def load(self):
		'''
		:return: sample of X, y
		'''
		old_pointer = self.pointer
		self.pointer = (self.pointer + self.batch_size) % len(self.X)

		if old_pointer >= self.pointer:
			# wrap around
			self.pointer = 0
			return np.array(self.X[old_pointer:]), np.array(self.y[old_pointer:])

		return np.array(self.X[old_pointer:self.pointer]), np.array(self.y[old_pointer:self.pointer])