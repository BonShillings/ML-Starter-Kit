import pickle
import numpy as np

class AbstractLoader():

	def __init__(self, data_source, batch_size):
		self.data_source = data_source
		self.batch_size = batch_size

	def load(self):
		pass
