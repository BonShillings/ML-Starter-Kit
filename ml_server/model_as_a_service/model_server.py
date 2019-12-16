import json

from ml_kit.models.linear_model import LinearModel
from ml_kit.evaluators.batch_evaluator import BatchEvaluator
from ml_kit.loaders.list_loader import ListLoader

from flask import Flask, flash, redirect, render_template, request, session, abort

AVAILABLE_MODELS = {"linear" : LinearModel}

LOADERS = {"list_loader": ListLoader}

EVALUATORS = {"batch_evaluate": BatchEvaluator}

MODEL = None

@app.route("/")
def index():
	return "A Simple Model Server!"

@app.route("/load", methods=['POST'])
def load():

	model_config = request.args.config
	model_path = request.args.path
	model_name = request.args.model

	MODEL = AVAILABLE_MODELS[model_name](**model_config).load(model_path)

	return "Model {} loaded from path {}".format(model_path, model_name)


@app.route("/evaluate", methods=['POST'])
def evaluate():

	if MODEL is None:
		return "Model not loaded"

	batch = json.loads(request.args.batch)
	batch_size = request.args.batchsize
	loader = request.args.loader
	evaluator = request.args.evaluator
	device = request.args.device

	loader = LOADERS[loader](batch, batch_size)
	evaluator = EVALUATORS[evaluator](MODEL, device)
	result = evaluator.evaluate(loader, 1)

	return result

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=82)