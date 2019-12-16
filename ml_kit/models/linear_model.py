import torch
import torch.nn as nn

class LinearModel(nn.Module):

	def __init__(self, input_size, embedding_size, output_size, device):
		super(LinearModel, self).__init__()

		# define layers
		self.device = device

		self.embedding = nn.Linear(input_size, embedding_size).to(self.device)
		self.output = nn.Linear(embedding_size, output_size).to(self.device)

	def forward(self, batch):
		embeddings = self.embedding(batch)
		return self.output(embeddings), embeddings

	def save(self,path):
		torch.save(self.state_dict(), path)

	def load(self, path):
		self.load_state_dict(torch.load(path))