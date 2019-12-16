import json
from estimates_server.mongo_db.mongo_estimates import MongoEstimates

from flask import Flask, flash, redirect, render_template, request, session, abort

SERVERS = {"MongoEstimates": MongoEstimates}

SERVER = None

@app.route("/")
def index():
	return "A Simple Estimate Vendor!"

@app.route("/connect", methods=['POST'])
def connect():

	server_name = request.args.server
	SERVER = SERVERS[server_name]()
	return "Estimate server connected"


@app.route("/load", methods=['POST'])
def load():

	if SERVER is None:
		return "Server is not initialised"

	estimates = json.loads(request.args.estimates)
	SERVER.load(estimates)
	return "Server estimates loaded"


@app.route("/evaluate", methods=['POST'])
def evaluate():

	if SERVER is None:
		return "Server is not initialised"

	key = request.args.key
	estimate = SERVER.lookup(key)

	if not estimate:
		return "Key {} not found".format(key)

	return estimate

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=80)