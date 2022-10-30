import pickle

import numpy as np
import torch

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

dataset_features = np.load('/home/usuario/git-repos/CC6409-Flask/classifier-app/dataset_features.npy')
dataset_features = torch.from_numpy(dataset_features)

with open('/home/usuario/git-repos/CC6409-Flask/classifier-app/img_names.pickle', 'rb') as data:
    img_names = pickle.load(data)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_closest_match(features):
    # ...encontrar la imagen más similar del conjunto objetivo...
    # ...retornar su nombre de archivo...

    euclidean_dist = (torch.from_numpy(dataset_features) - features).pow(2).sum(1).sqrt()
    idx_min_dist = euclidean_dist.topk(3, dim=0, largest=False).indices
    closest_match = [img_names[k] for k in list(idx_min_dist.reshape(-1))]

    return closest_match
