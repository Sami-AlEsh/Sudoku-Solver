import cv2 as cv
import numpy as np
from Dataset import Generator


def _get_trained_model(train, train_labels, knn=None):
    if knn is None:
        knn = cv.ml.KNearest_create()
    knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
    return knn

def get_trained_knn():
    # Generate numbers dataset
    m_dataset = Generator.generate_numbers_dataset()

    # Generate features:
    features_set = []
    winSize = (32, 32)
    blockSize = (8, 8)
    blockStride = (8, 8)
    cellSize = (4, 4)
    nbins = 9
    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    for digits_row in m_dataset:
        for digit in digits_row:
            # Convert image from float32 -> uint8 for HOG
            digit8bit = cv.normalize(digit, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            # show('d', digit8bit)

            # Applying HOG (Return array by length of 576)
            h = hog.compute(digit8bit)
            features_set.append(h)

    features_set = np.array(features_set).reshape(-1, 576).astype(np.float32)
    # [DEBUG] print('dataset features shape:', features_set.shape)

    # Training:
    trainset = features_set
    labels = np.arange(1, 10)
    train_labels = []
    for ls in [labels] * m_dataset.shape[0]:
        for item in ls:
            train_labels.append([item])
    return _get_trained_model(trainset, np.array(train_labels))
