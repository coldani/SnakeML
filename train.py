import argparse
import pickle

from genetic_algorithm import GeneticAlgorithm

# PARSE ARGUMENTS

parser = argparse.ArgumentParser(description='Training of Snake')
parser.add_argument(
    '--size', '-s', type=int, dest='pop_size',
    help="Population size at each epoch")

parser.add_argument(
    '--epochs', '-e', type=int, dest='epochs',
    help="Number of epochs for training")

parser.add_argument(
    '--matches', '-m', type=int, nargs='?', dest='matches',
    help="Number of matches played by each individual to calculate fitness",
    default=1)

args = parser.parse_args()
pop_size = args.pop_size
num_epochs = args.epochs
num_matches = args.matches

# END OF ARGUMENTS PARSING


model = GeneticAlgorithm(pop_size, [10, 10, 10])
model.train(num_epochs, num_matches, print_frequency=1)

with open("saved_models/model.pickle", "wb") as f:
    pickle.dump(model, f)
