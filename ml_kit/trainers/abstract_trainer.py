class AbstractTrainer:

	def __init__(self, model, criterion, optimizer, device):
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.device = device


	def train(self, loader, num_batches):
		'''
		Trains for 1 epoch (defined by batch_iterations)
		:param loader: AbstractLoader Implementation
		:param num_batches: number of batches to call
		:return: training loss
		'''
		pass

	def test(self, loader, num_batches):
		'''
		Test dataset
		:param loader: AbstractLoader Implementation
		:param num_batches: number of batches to test on
		:return: loss, predictions
		'''
		pass