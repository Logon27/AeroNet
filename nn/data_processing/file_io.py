import dill
from nn import Network

#Provides the ability to save and load neural network objects from files.
#The entire network is saved as an object to a file.

def save_network(network: Network, file_path: str):
    with open(file_path, "wb") as file:
        print("Saving network to file {}...".format(file_path))
        dill.dump(network, file, dill.HIGHEST_PROTOCOL)
        print("Save complete.")

def load_network(file_path: str) -> Network:
    with open(file_path, "rb") as file:
        print("Loading network from file {}...".format(file_path))
        network = dill.load(file)
        print("Loading complete.")
        return network