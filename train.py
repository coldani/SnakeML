from genetic_algorithm import GeneticAlgorithm
import pickle

model = GeneticAlgorithm(1000, [10, 10, 10])
model.train(100, 3, 200, 1)

with open("saved_models/model.pickle", "wb") as f:
    pickle.dump(model, f)
