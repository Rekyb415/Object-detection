# coding: utf-8
import pickle

def save_params(self, file_name="params.pkl"):
    params = {}
    for key, val in self.params.items():
        params[key] = val
    with open(file_name, 'wb') as f:
        pickle.dump(params, f)

def load_params(self, file_name="params.pkl"):
    with open(file_name, 'rb') as f:
        params = pickle.load(f)
    for key, val in params.items():
        self.params[key] = val

    for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
        self.layers[layer_idx].W = self.params['W' + str(i+1)]
        self.layers[layer_idx].b = self.params['b' + str(i+1)]
