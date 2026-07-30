import pickle

with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

with open("human_vocab.pkl", "rb") as f:
    human_vocab = pickle.load(f)

with open("machine_vocab.pkl", "rb") as f:
    machine_vocab = pickle.load(f)

with open("inv_machine_vocab.pkl", "rb") as f:
    inv_machine_vocab = pickle.load(f)

print(dataset)