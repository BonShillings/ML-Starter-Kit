import pymongo

class MongoEstimates:

	def __init__(self, estimates=None):
		'''
		list of dictionary objects
		:param estimates:
		'''
		self.client = pymongo.MongoClient("mongodb://localhost:27017/")
		self.db = self.client["estimates_database"]
		self.table = self.db["estimates"]

		if estimates:
			self.load(estimates)

	def load(self, estimates):
		self.table.insert_many(estimates)

	def lookup(self, key):

		query = {"key": key}
		results = self.table.find(query, {"estimate": 1})

		return results[0]