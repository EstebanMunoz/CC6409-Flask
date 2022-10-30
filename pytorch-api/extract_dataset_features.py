import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from app import feature_extractor
from utils import transform_image


def get_size_feature_extractor(feature_extractor):
    out = feature_extractor(torch.rand(1, 3, 224, 224))
    size = list(out.values())[0].shape
    return size


def get_dataset_features(dataset):
    loader = DataLoader(dataset, batch_size=1)

    extractor_size = get_size_feature_extractor(feature_extractor)
    size = (len(dataset), *extractor_size[1:])
    dataset_features = torch.empty(size)

    for i, batch in enumerate(loader):
        batch_features = feature_extractor(batch)
        dataset_features[i] = list(batch_features.values())[0]

    return dataset_features.detach().numpy()


class CatalogImages(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        with open(img_path, 'rb') as file:
            image = file.read()

        if self.transform:
            image = self.transform(image)

        return image


if __name__ == '__main__':
    Catalog = CatalogImages('/media/disco-compartido/mc4/catalogo', transform_image)
    dataset_features = get_dataset_features(Catalog)
    np.save('/home/usuario/git-repos/CC6409-Flask/classifier-app/dataset_features', dataset_features)
    with open('/home/usuario/git-repos/CC6409-Flask/classifier-app/img_labels.pickle', 'wb') as output:
        pickle.dump(Catalog.img_names, output)