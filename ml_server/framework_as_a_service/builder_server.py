import json

from ml_kit.models.linear_model import LinearModel
from ml_kit.loaders.tabular_file_loader import TabularFileLoader
from ml_kit.trainers.batch_trainer import BatchTrainer

from flask import Flask, flash, redirect, render_template, request, session, abort

app = Flask(__name__)

AVAILABLE_MODELS = {"linear" : LinearModel}

LOADERS = {
	"file_loader": TabularFileLoader
}

TRAINERS = {"batch_trainer", BatchTrainer}

MODEL = None

@app.route("/")
def index():
	return "A Simple Model Builder!"

@app.route("/initialise", methods=['POST'])
def initialise():
	model_name = request.args.model
	config = json.loads(request.args.config)

	MODEL = AVAILABLE_MODELS[model_name](**config)

	return "MODEL {} is initialise".format(model_name)

@app.route("/train", methods=['POST'])
def train():

	loader = request.args.loader
	trainer = request.args.trainer
	data_source = request.args.data
	batch_size = request.args.batchsize
	num_batches = request.args.batches

	if data_source not in LOADERS:
		pass

	loader = LOADERS[loader](data_source, batch_size)

	loss = TRAINERS[trainer].train(loader, num_batches)

	return "Model trained to loss of {}".format(loss)

@app.route("/save", methods=['POST'])
def save():

	if not MODEL:
		return "Model is not initialised"

	MODEL.save(request.args.path)

	return "Active MODEL {} has been saved to {}".format(MODEL, request.args.path)


@app.route("/load", methods=['POST'])
def load():
	if not MODEL:
		return "Model is not initialised"

	MODEL.load(request.args.path)

	return "Active MODEL {} has been loaded from {}".format(MODEL, request.args.path)

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=81)