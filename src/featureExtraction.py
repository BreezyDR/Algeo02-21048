import cv2
import numpy as np
# import random
import os
import matplotlib.pyplot as plt
import pickle

from typing import List


# Feature extractor
def extract_features(image_path : str, vector_size : int = 32) -> List[int]:
    image = cv2.imread(image_path)
    # regarding mode='RGB',
    # 'mode' parameter is a scipy.misc.imread specific parameter
    # which imread() method is removed on the 1.2.0 version of scipy
    # the alternatives would be imageio.imread
    # but i'll try using cv2's imread for now

    try:
        alg = cv2.KAZE_create()

        kps = alg.detect(image)

        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]

        kps, dsc = alg.compute(image, kps)

        # dsc = dsc.flatten()
 
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print('Error: ', e)
        return None
    return dsc


def batch_extractor(images_path : str, pickled_db_path : str ="features.pck") -> None :
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    result = {}
    for f in files:
        print('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f)

        print(np.array(result[name]).shape, name, 'sizeof')

    with open(pickled_db_path, 'wb') as fp:
        # 'wb' is used instead of 'w' since it produces an error
        # see : https://stackoverflow.com/questions/13906623/using-pickle-dump-typeerror-must-be-str-not-bytes
        pickle.dump(result, fp)


# show an image with matplotlib
# caution: opencv color format is BGR, instead of plt's RGB, hence the need for cvtColor
def show_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()