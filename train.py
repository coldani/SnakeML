import argparse
import pickle

from genetic_algorithm import GeneticAlgorithm

# wrap inside block to allow for multiprocessing to work correctly
if __name__ == '__main__':
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

    parser.add_argument(
        '--gain', '-g', type=int, nargs='?', dest='life_gain',
        help="Snake life gain whenever an apple is eaten",
        default=200)

    parser.add_argument(
        '--layers', '-l', type=int, nargs='+', dest='layers_size',
        help="Number of neurons in each of the neural network layers, excluding the output layer (which has 3 neurons by design)",
        default=[10, 10, 10])

    parser.add_argument(
        '--length', '-i', type=int, nargs='?', dest='initial_length',
        help="Initial snake length",
        default=1)

    args = parser.parse_args()
    pop_size = args.pop_size
    num_epochs = args.epochs
    num_matches = args.matches
    life_gain = args.life_gain
    layers_size = args.layers_size
    initial_length = args.initial_length

    # END OF ARGUMENTS PARSING

    model = GeneticAlgorithm(pop_size, layers_size)
    model.train(
        num_epochs, num_matches, snake_life_gain=life_gain,
        snake_length=initial_length, print_frequency=1, multiprocessing=True)

    layers_str = '_'.join(str(x) for x in model.layers_size)
    name = f"{layers_str}_s{pop_size}_e{num_epochs}_m{num_matches}_g{life_gain}_i{initial_length}"
    with open(f"saved_models/{name}.pickle", "wb") as f:
        pickle.dump(model, f)
