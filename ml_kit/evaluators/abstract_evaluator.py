class AbstractEvaluator:
	def __init__(self, model, device):
		self.model = model
		self.device = device

	def evaluate(self, loader, num_batches):
		'''
		:param loader:
		:return:
		'''
		pass
