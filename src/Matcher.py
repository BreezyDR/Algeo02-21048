import scipy
import numpy as np
from src.featureExtraction import extract_features
import pickle

class Matcher(object):

    def __init__(self, pickled_db_path="features.pck"):
        with open(pickled_db_path, 'rb') as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)

    def cos_cdist(self, vector):
        # getting cosine distance between search image and images database
        # v = vector.reshape(1, -1)
        print(vector.shape, self.matrix.shape)
        return [scipy.spatial.distance.cdist(self.matrix[i].reshape(1, -1), vector.reshape(1, -1), 'cosine') for i in range(3)]

    def match(self, image_path, topn=3):
        features = extract_features(image_path)
        print(np.array(features).shape)
        img_distances = self.cos_cdist(features)
        print('disctp', np.array(img_distances).shape)
        # getting top 5 records
        nearest_ids = np.argsort(img_distances)[:topn].tolist()
        print(np.array(nearest_ids).shape, np.array(self.names).shape)
        
        nearest_img_paths = self.names[nearest_ids].tolist()
        print(nearest_ids, nearest_img_paths)
        print(np.array(nearest_ids).reshape(-1, 1), np.array(nearest_img_paths).reshape(-1, 1))

        return nearest_img_paths, img_distances[nearest_ids].tolist() #nyerah disini