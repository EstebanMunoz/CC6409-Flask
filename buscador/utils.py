import pickle
import os

from dotenv import load_dotenv
import numpy as np
import torch

load_dotenv()

FEATURES_PATH = os.getenv('FEATURES_PATH')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

dataset_features = np.load(f'{FEATURES_PATH}/dataset_features.npy')
dataset_features = torch.from_numpy(dataset_features)

with open(f'{FEATURES_PATH}/img_names.pickle', 'rb') as data:
    img_names = pickle.load(data)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_closest_match(features):
    euclidean_dist = (dataset_features - features).pow(2).sum(1).sqrt()
    idx_min_dist = euclidean_dist.topk(3, dim=0, largest=False).indices
    closest_match = [img_names[k] for k in list(idx_min_dist.reshape(-1))]

    return closest_match
