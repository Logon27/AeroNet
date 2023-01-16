import dill
from nn import Network

#Provides the ability to save and load neural network objects from files.
#The entire network is saved as an object to a file.

def save_network(network: Network, filePath: str):
    with open(filePath, "wb") as file:
        print("Saving network to file {}...".format(filePath))
        dill.dump(network, file, dill.HIGHEST_PROTOCOL)
        print("Save complete.")

def load_network(filePath: str) -> Network:
    with open(filePath, "rb") as file:
        print("Loading network from file {}...".format(filePath))
        network = dill.load(file)
        print("Loading complete.")
        return network