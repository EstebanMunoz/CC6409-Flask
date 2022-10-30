import os
import pickle

import numpy as np
import torch

from utils import get_features


# def get_size_feature_extractor(feature_extractor):
#     out = feature_extractor(torch.rand(1, 3, 224, 224))
#     size = list(out.values())[0].shape
#     return size


# def get_dataset_features(dataset):
#     loader = DataLoader(dataset, batch_size=1)

#     extractor_size = get_size_feature_extractor(feature_extractor)
#     size = (len(dataset), *extractor_size[1:])
#     dataset_features = torch.empty(size)

#     for i, batch in enumerate(loader):
#         batch_features = feature_extractor(batch)
#         dataset_features[i] = list(batch_features.values())[0]

#     return dataset_features.detach().numpy()


# class CatalogImages(Dataset):
#     def __init__(self, img_dir, transform=None):
#         self.img_dir = img_dir
#         self.img_names = sorted(os.listdir(img_dir))
#         self.transform = transform

#     def __len__(self):
#         return len(self.img_names)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_names[idx])
#         with open(img_path, 'rb') as file:
#             image = file.read()

#         if self.transform:
#             image = self.transform(image)

#         return image


def get_dataset_features(catalog_path):
    img_dirs = sorted(os.listdir(catalog_path))
    dataset_features = torch.empty(len(img_dirs), 1024, 1, 1)

    for i, img_name in enumerate(img_dirs):
        img_path = os.path.join(catalog_path, img_name)
        with open(img_path, 'rb') as file:
            img = file.read()
        img_features = get_features(img)
        dataset_features[i] = img_features
        print(f"features extracted from image {i+1}")
    
    return dataset_features.detach().numpy(), img_dirs


if __name__ == '__main__':
    # Catalog = CatalogImages('pytorch-api/catalogo', transform_image)
    # dataset_features = get_dataset_features(Catalog)
    dataset_features, img_names = get_dataset_features("/media/disco-compartido/mc4/catalogo")
    np.save('classifier-app/dataset_features', dataset_features)
    with open('classifier-app/img_labels.pickle', 'wb') as output:
        pickle.dump(img_names, output)
