def predict(evaluator, loader):
	predictions, _ = evaluator.evaluate(loader, loader.batches_in_data())
	return predictions