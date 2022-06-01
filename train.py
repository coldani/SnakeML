from genetic_algorithm import GeneticAlgorithm
import pickle

model = GeneticAlgorithm(10, [10, 10, 10])  # 1000
model.train(2, 3, 200, 1)  # 200

with open("saved_models/model.pickle", "wb") as f:
    pickle.dump(model, f)
