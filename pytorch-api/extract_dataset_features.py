import os
import pickle

import numpy as np
from torch import cat
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from app import feature_extractor
from utils import transform_image


def get_dataset_features(dataset, batch_size=1):
    loader = DataLoader(dataset, batch_size=batch_size)
    list_features = []
    for batch in loader:
        batch_features = feature_extractor(batch)
        list_features += list(batch_features.values())

    dataset_features = cat(list_features, dim=0)
    return dataset_features.detach().numpy()


class CatalogImages(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        with open(img_path, 'rb') as file:
            image = file.read()

        if self.transform:
            image = self.transform(image)

        return image


if __name__ == '__main__':
    Catalog = CatalogImages('/home/usuario/git-repos/CC6409-Flask/pytorch-api/catalogo', transform_image)
    dataset_features = get_dataset_features(Catalog, batch_size=460)
    np.save('/home/usuario/git-repos/CC6409-Flask/classifier-app/dataset_features', dataset_features)
    with open('/home/usuario/git-repos/CC6409-Flask/classifier-app/img_labels.pickle', 'wb') as output:
        pickle.dump(Catalog.img_names, output)