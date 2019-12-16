import logging
import sys
import torch

from metrics.metrics import Metrics

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

logger.setLevel(logging.DEBUG)

def train_model(loaders, trainer, device, epochs):

	train_loader = loaders[0]
	validation_loader = loaders[1]
	test_loader = loaders[2]

	metrics = Metrics()

	for epoch in range(epochs):
		trainer.train(train_loader, train_loader.batches_in_data())

		train_loss, train_pred, train_embeddings = trainer.test(train_loader, train_loader.batches_in_data())
		val_loss, val_pred, val_embeddings = trainer.test(validation_loader, validation_loader.batches_in_data())
		test_loss, test_pred, test_embeddings = trainer.test(test_loader, test_loader.batches_in_data())

		train_labels = torch.as_tensor(train_loader.y, dtype=torch.float, device=device)
		val_labels = torch.as_tensor(validation_loader.y, dtype=torch.float, device=device)
		test_labels = torch.as_tensor(test_loader.y, dtype=torch.float, device=device)

		evaluate_predictions(epoch, 'train', train_loss, train_pred, train_labels, metrics)
		evaluate_predictions(epoch, 'validation', val_loss, val_pred, val_labels, metrics)
		evaluate_predictions(epoch, 'test', test_loss, test_pred, test_labels, metrics)


def evaluate_predictions(epoch, split, loss, predictions, labels, metrics):
	'''
	:param split: train, val or test
	'''
	pearson = metrics.pearson(predictions, labels)
	mse = metrics.mse(predictions, labels)
	accuracy = metrics.accuracy(predictions, labels)
	logger.info('==> Epoch {}, {} \tLoss: {}\tAccuracy: {}\tPearson: {}\tMSE: {}'.format(
		epoch, split, loss, accuracy, pearson, mse))