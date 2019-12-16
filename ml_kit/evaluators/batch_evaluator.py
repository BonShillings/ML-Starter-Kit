''' Copyright (C) Sean Billings - All Rights Reserved
 	Unauthorized copying of this file, via any medium is strictly prohibited
 	Proprietary and confidential
 	Written by Sean Billings <s.a.f.billings@gmail.com>, August 2019
'''

import torch
from tqdm import tqdm
from ml_kit.evaluators.abstract_evaluator import AbstractEvaluator

class BatchEvaluator(AbstractEvaluator):

	def __init__(self, model, device):
		super(BatchEvaluator).__init__(model, device)

	def evaluate(self, loader, num_batches):
		self.model.eval()
		with torch.no_grad():
			predictions = []
			embeddings = []

			for _ in tqdm(range(num_batches), desc='Testing batches ' + str(num_batches) + ''):
				outputs, batch_embeddings = get_batch_output_and_embeddings(self.model, loader)

				for output in outputs:
					predictions.append(torch.argmax(output).item())

				for embedding in batch_embeddings:
					embeddings.append(embedding)

			return torch.as_tensor(predictions, dtype=torch.float, device=self.device), embeddings

def get_batch_output_and_embeddings(self, model, loader):
	batch, targets = loader.load()

	if loader.use_token_embeddings:
		batch = torch.as_tensor(batch, dtype=torch.long).to(self.device)
	else:
		batch = torch.as_tensor(batch, dtype=torch.float).to(self.device)

	outputs, embeddings = model(batch)

	return outputs, embeddings

