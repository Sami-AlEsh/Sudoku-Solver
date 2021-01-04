import ModelsUtil
import cv2 as cv
import numpy as np

from Utils import show


def generate_numbers_dataset():
    """Generates grayscale digits images dataset"""
    # Font Properties:
    photoSize = (26, 18)
    fontScale = 1
    thickness = [1, 2, 3]

    # Fonts:
    fonts = [f for f in range(8)]
    fonts.remove(1), fonts.remove(5), fonts.remove(7)
    print('generating dataset for', len(fonts), 'fonts with', len(thickness), 'different thicknesses...')

    dataset = []
    for font in fonts:
        for thick in thickness:
            numbers_gray = []
            # [DEBUG] numbers_image = []

            # Generating all numbers:
            for number in [num for num in range(1, 10)]:
                photo = np.zeros((photoSize[0], photoSize[1], 3), np.float32)
                digit_BGR = cv.putText(photo, str(number), org=(-2, photoSize[0] - 3), fontFace=font,
                                       fontScale=fontScale, color=(255, 255, 255), thickness=thick,
                                       lineType=cv.LINE_AA)
                digit_gray = cv.cvtColor(digit_BGR, cv.COLOR_BGR2GRAY)

                # Make image square
                side = np.zeros((26, 4))
                digit_gray = np.hstack((np.hstack((side, digit_gray)), side))

                # Scale image to 32x32 for HOG
                digit_gray = cv.resize(digit_gray, (32, 32))

                # Add number to gray list:
                numbers_gray.append(digit_gray)

                # [DEBUG] Add number to NumbersImage
                # if len(numbers_image) > 0:
                #     numbers_image = np.hstack((numbers_image, digit_gray))
                # else:
                #     numbers_image = digit_gray

            # Add digits gray to train set list
            dataset.append(numbers_gray)

            # [DEBUG]Show Numbers
            # cv.imshow('Numbers', numbers_image)
            # cv.waitKey(0)

    # add preloaded font to dataset
    def get_fonts_images(j):
        nums = []
        for i in range(1, 10):
            path = f'Dataset/fonts/Font{j}/{i}.jpg'
            num = cv.imread(path, 0)
            num = cv.resize(num, (32, 32))
            nums.append(num)
        return nums

    folders_count = 35
    print('generating dataset for', folders_count, 'fonts...')
    for folder_id in range(1, folders_count+1):
        frame_9nums = get_fonts_images(folder_id)
        dataset.append(frame_9nums)

    dataset = np.array(dataset)
    print('generated dataset shape:', dataset.shape)
    return dataset


# Image itself as a feature
def generator_demo1():
    """Demo1: split data to train/test , then train KNN model on trainset then validate using test data. Features
    here are the image itself"""
    m_dataset = generate_numbers_dataset()
    trainset = m_dataset[:-1]
    testset = m_dataset[-1]
    print(trainset.shape)
    print(testset.shape)

    # Generate features:
    trainset = trainset.reshape(-1, 468).astype(np.float32)
    print('train data:', trainset.shape)
    testset = testset.reshape(-1, 468).astype(np.float32)
    print('test data:', testset.shape)

    # Testing:
    labels = np.arange(1, 10)
    train_labels = []
    for ls in [labels] * 14:
        for item in ls:
            train_labels.append([item])

    knn = ModelsUtil.get_trained_model(trainset, np.array(train_labels))
    ret, result, neighbours, dist = knn.findNearest(np.array(testset), k=5)
    print('result:', result.reshape(-1, 9)[0])

# SIFT IS NOT WORKING VERY WELL
def generator_demo2():
    """Demo1: split data to train/test , then train KNN model on trainset then validate using test data. Features here
    are extracted from image using SIFT"""
    m_dataset = generate_numbers_dataset()
    trainset = m_dataset[:-1]
    testset = m_dataset[-1]
    print(trainset.shape)
    print(testset.shape)

    # Generate features:
    sift = cv.SIFT_create()
    for digits_row in trainset:
        for digit in digits_row:
            # Convert image from float32 -> uint8 for SIFT
            digit8bit = cv.normalize(digit, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            # Applying SIFT
            kp, des = sift.detectAndCompute(digit8bit, None)
            if des is not None:
                print('des type:', type(des), 'and length:', len(des[0]))
                print(des)
                img = cv.drawKeypoints(digit8bit, kp, digit)
                cv.imshow('sift', img)
                cv.waitKey(0)
    trainset = trainset.reshape(-1, 468).astype(np.float32)
    print('train data:', trainset.shape)
    testset = testset.reshape(-1, 468).astype(np.float32)
    print('test data:', testset.shape)

    # Testing:
    labels = np.arange(1, 10)
    train_labels = []
    for ls in [labels] * 14:
        for item in ls:
            train_labels.append([item])

    knn = ModelsUtil.get_trained_model(trainset, np.array(train_labels))
    ret, result, neighbours, dist = knn.findNearest(np.array(testset), k=5)
    print('result:', result.reshape(-1, 9)[0])

# HOG as image features
def generator_demo3():
    """Demo2: split data to train/test , then train KNN model on trainset then validate using test data. Features here
    are extracted from image using HOG"""
    m_dataset = generate_numbers_dataset()

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

            # Applying HOG (Return array by length of 576)
            h = hog.compute(digit8bit)
            features_set.append(h)

    features_set = np.array(features_set).reshape(-1, 576).astype(np.float32)
    print('dataset features shape:', features_set.shape)

    trainset = features_set[:-9]
    testset = features_set[-9:]
    print('train set:', trainset.shape)
    print('test set:', testset.shape)

    # Testing:
    labels = np.arange(1, 10)
    train_labels = []
    for ls in [labels] * 14:
        for item in ls:
            train_labels.append([item])

    knn = ModelsUtil.get_trained_model(trainset, np.array(train_labels))
    ret, result, neighbours, dist = knn.findNearest(np.array(testset), k=5)
    print('result:', result.reshape(-1, 9)[0])


# [DEBUG] Show Demos:
# generator_demo3()
# generator_demo1()