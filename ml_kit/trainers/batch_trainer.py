''' Copyright (C) Sean Billings - All Rights Reserved
 	Unauthorized copying of this file, via any medium is strictly prohibited
 	Proprietary and confidential
 	Written by Sean Billings <s.a.f.billings@gmail.com>, August 2019
'''

import torch

from tqdm import tqdm
from ml_kit.trainers.abstract_trainer import AbstractTrainer

class BatchTrainer(AbstractTrainer):
	def __init__(self, model, criterion, optimizer, device):
		super(BatchTrainer).__init__(model, criterion, optimizer, device)
		self.batch = 0
		self.epoch = 0

	# helper function for training
	def train(self, loader, num_batches):
		'''
        :param loader: AbstractLoader Implementation
        :param num_batches: number of batches to call
        :return:
        '''

		self.model.train()
		self.optimizer.zero_grad()
		total_loss = 0.0
		for i in tqdm(range(num_batches), desc='Training Epoch ' + str(self.epoch) + ''):

			outputs, loss, _ = self.get_model_output(loader)

			total_loss += loss.item()

			loss.backward()
			self.optimizer.step()
			self.optimizer.zero_grad()
			self.batch += 1
		self.epoch += 1

		total_loss /= loader.batch_size * num_batches
		return total_loss

	def test(self, loader, num_batches):
		self.model.eval()
		with torch.no_grad():
			total_loss = 0.0
			predictions = []
			embeddings = []

			for _ in tqdm(range(num_batches), desc='Testing batches ' + str(num_batches) + ''):
				outputs, loss, batch_embeddings = self.get_model_output(loader)
				total_loss += loss.item()

				# outputs = outputs.squeeze().to(self.device)
				for output in outputs:
					predictions.append(torch.argmax(output).item())

				for embedding in batch_embeddings:
					embeddings.append(embedding)
			total_loss /= loader.batch_size * num_batches
			predictions = torch.as_tensor(predictions, dtype=torch.float, device=self.device)
			return total_loss, predictions, embeddings

	def get_model_output(self, loader):
		batch, targets = loader.load()

		batch = torch.as_tensor(batch, dtype=torch.float).to(self.device)

		targets = torch.as_tensor(targets, dtype=torch.long).to(self.device)
		outputs, embeddings = self.model(batch)

		loss = self.criterion(outputs, targets)
		return outputs, loss, embeddings