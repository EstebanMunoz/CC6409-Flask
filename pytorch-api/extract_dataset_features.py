import os
import pickle

import numpy as np

from utils import get_features


def get_dataset_features(catalog_path):
    img_dirs = sorted(os.listdir(catalog_path))
    dataset_features = np.empty((len(img_dirs), 1024, 1, 1))

    for i, img_name in enumerate(img_dirs):
        img_path = os.path.join(catalog_path, img_name)
        with open(img_path, 'rb') as file:
            img = file.read()
        img_features = get_features(img)
        dataset_features[i] = img_features.detach().numpy()
        print(f"features extracted from image {i+1}")
    
    return dataset_features, img_dirs


if __name__ == '__main__':
    dataset_features, img_names = get_dataset_features('/media/disco-compartido/mc4/catalogo')
    np.save('/home/usuario/git-repos/CC6409-Flask/classifier-app/dataset_features', dataset_features)
    with open('/home/usuario/git-repos/CC6409-Flask/classifier-app/img_names.pickle', 'wb') as dir:
        pickle.dump(img_names, dir)
