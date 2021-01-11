import cv2 as cv
import numpy as np


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
