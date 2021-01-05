import cv2 as cv
import numpy as np
from Utils import show
from Dataset import Generator


def get_opencv_train_test_data():
    gray = cv.imread('Dataset/opencv samples/digits.png', cv.IMREAD_GRAYSCALE)

    # Split image to 5000 cells, each 20x20 size
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    # Make it into a Numpy array: its size will be (50,100,20,20)
    cells = np.array(cells)

    # Now we prepare the training data and test data
    train = cells[:, :50].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)
    test = cells[:, 50:100].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)

    # Create labels for train and test data
    k = np.arange(10)
    train_labels = np.repeat(k, 250)[:, np.newaxis]
    test_labels = train_labels.copy()

    return train, train_labels, test, test_labels


def get_trained_model(train, train_labels, knn=None):
    if knn is None:
        knn = cv.ml.KNearest_create()
    knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
    return knn


def calculate_accuracy(result, test_labels):
    matches = result == test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct * 100.0 / result.size
    return accuracy


def save_train_data(train, train_labels, path):
    # Reduce file size by converting type to np.uint8
    train = train.astype(np.uint8)
    np.savez(path, train=train, train_labels=train_labels)


def load_trained_data(path):
    with np.load(path) as data:
        print('loading data', data.files)
        # Revert data type to np.float32
        train = data['train'].astype(np.float32)
        train_labels = data['train_labels']
    return train, train_labels


# Usage:
# train, train_labels, test, test_labels = get_opencv_train_test_data()
# # save_train_data('knn_data.npz', train, train_labels)
# # train, train_labels = load_trained_data('knn_data.npz')
# knn = get_trained_model(train, train_labels)
# # detect digit:
# ret, result, neighbours, dist = knn.findNearest(test, k=5)
# print(calculate_accuracy(result, test_labels))


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

    # Add hand written digit dataset
    # _train, _train_labels, _test, _test_labels = get_opencv_train_test_data()
    # knn = get_trained_model(_train, _train_labels)
    return get_trained_model(trainset, np.array(train_labels))


def get_trained_svm():
    def deskew(img, SZ=20, affine_flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR):
        """Before finding the HOG, we deskew the image using its second order moments. So we first define a function
        deskew() which takes a digit image and deskew it"""
        m = cv.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11'] / m['mu02']
        M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
        img = cv.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
        return img

    def hog(img, bin_n=16):
        gx = cv.Sobel(img, cv.CV_32F, 1, 0)
        gy = cv.Sobel(img, cv.CV_32F, 0, 1)
        mag, ang = cv.cartToPolar(gx, gy)

        # quantizing binvalues in (0...16)
        bins = np.int32(bin_n * ang / (2 * np.pi))

        # Divide to 4 sub-squares
        bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)
        return hist

    ######     Now Training      ########################
    # Generate numbers dataset
    m_dataset = Generator.generate_numbers_dataset()

    # Make image square
    train_cells = []
    for digits_row in m_dataset:
        train_cells_row = []
        for digit in digits_row:
            digit = digit.reshape((32, 32, 1)).astype(np.float32)

            train_cells_row.append(digit)
        train_cells.append(train_cells_row)

    data = []
    deskewed = []
    hogdata = []
    for row in train_cells:
        for d in row:
            data.append(d)
            _ = deskew(d)
            deskewed.append(_)
            hogdata.append(hog(_))

    # data = np.float32(data).reshape((-1, 1024))
    trainData = np.float32(hogdata).reshape(-1, 64)
    responses = np.array([i for i in range(1, 10)] * m_dataset.shape[0])

    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setGamma(5.383)
    svm.setC(2.67)

    svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
    # svm.save('Modles/svm_data.dat')
    return svm