import torch

class BaseModel(torch.nn.Module):
	def __init__(self):
		pass

	def forward(self, *args):
		'''
		:param args:
		:return: outputs (predictions, embeddings)
		'''
		pass

	def save(self, path):
		pass

	def load(self, path):
		pass
