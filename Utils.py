import cv2 as cv
import numpy as np


def show(name, image, pause=True):
    cv.imshow(name, image)
    if pause:
        cv.waitKey(0)
        cv.destroyWindow(name)


clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
def equalize_image(image, mode):
    if mode == 'HIST':
        return cv.equalizeHist(image)
    elif mode == 'CLAHE':  # CLAHE: Contrast Limited Adaptive Histogram Equalization
        return clahe.apply(image)
    else:
        print(f'Unknown equlizer mode/{mode}/')
        return image


def get_grid_lines(gray_frame):
    def get_horizontal_lines(thresh_frame):
        horizontal = np.copy(thresh_frame)

        # Specify size on horizontal axis
        cols = horizontal.shape[1]
        horizontal_size = cols // 9

        # Create structure element (Kernel) for extracting horizontal lines
        horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))

        # Apply morphology operations
        horizontal = cv.morphologyEx(horizontal, cv.MORPH_OPEN, horizontalStructure)  # erode->dilate

        # Apply plus dilation (4, 4)
        horizontal = cv.dilate(horizontal, np.ones((4, 4), np.uint8))

        return horizontal

    def get_vertical_lines(thresh_frame):
        vertical = np.copy(thresh)

        # Specify size on vertical axis
        rows = vertical.shape[0]
        verticalsize = rows // 9

        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))

        # Apply morphology operations
        vertical = cv.morphologyEx(vertical, cv.MORPH_OPEN, verticalStructure)  # erode->dilate

        # Apply plus dilation (4, 4)
        vertical = cv.dilate(vertical, np.ones((4, 4), np.uint8))

        return vertical

    ############## END Inner functions ##############
    # thresh = cv.adaptiveThreshold(gray_frame, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, -2)
    thresh = cv.adaptiveThreshold(gray_frame, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 33, -1)
    return (get_horizontal_lines(thresh), get_vertical_lines(thresh))


def normalize_sudoko_board(square_gray_frame, fast):
    h, v = get_grid_lines(square_gray_frame)
    roi_grids = cv.bitwise_or(h, v)
    # [DEBUG] show('roi', roi_grids, False)
    # [DEBUG] show('H', h, False)
    # [DEBUG] show('V', v, False)

    roi = cv.bitwise_not(roi_grids)

    if fast:
        square_gray_frame = cv.GaussianBlur(square_gray_frame, (9, 9), 0)
    else:
        square_gray_frame = cv.bilateralFilter(square_gray_frame, 9, 60, 60)
    square_gray_frame = cv.adaptiveThreshold(square_gray_frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv.THRESH_BINARY_INV, 5, 2)
    # Remove Grid lines
    sudoko_board = cv.bitwise_and(roi, square_gray_frame)

    # Remove white-noise-points by opening
    sudoko_board = cv.morphologyEx(sudoko_board, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (2, 2)))
    sudoko_board = cv.morphologyEx(sudoko_board, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
    return sudoko_board


def find_number_bound(digit_img):
    # Note this functions assumes there is a number(white area) in image
    xs = digit_img.shape[1]
    xe = 0
    ys = digit_img.shape[0]
    ye = 0
    for i in range(digit_img.shape[1]):
        for j in range(digit_img.shape[0]):
            if 0 < digit_img[i][j] and i < xs:
                xs = i
            if 0 < digit_img[i][j] and i > xe:
                xe = i
            if 0 < digit_img[i][j] and j < ys:
                ys = j
            if 0 < digit_img[i][j] and j > ye:
                ye = j
    return xs, xe, ys, ye


def find_number_bound_enhanced(digit_img):
    # Note this functions assumes there is a number(white area) in image
    xs = digit_img.shape[1] - 1
    xe = 0
    ys = digit_img.shape[0] - 1
    ye = 0
    quarter = digit_img.shape[0] // 2  # 16
    side = digit_img.shape[0] - 1  # 31 (because counting from zero)
    for i in range(quarter):
        for j in range(quarter):
            # Get 4 points
            c1 = digit_img[i][j]
            c2 = digit_img[i][side - j]
            c3 = digit_img[side - i][j]
            c4 = digit_img[side - i][side - j]

            # Update Xs, Xe variables:
            i1 = i
            i2 = side - i
            i1_valid = c1 or c2 > 0
            i2_valid = c3 or c4 > 0
            if i1_valid and i1 < xs:
                xs = i1
            if i2_valid and i2 < xs:
                xs = i2
            if i1_valid and i1 > xe:
                xe = i1
            if i2_valid and i2 > xe:
                xe = i2

            # Update Ys, Ye variables:
            j1 = j
            j2 = side - j
            j1_valid = c1 or c3 > 0
            j2_valid = c2 or c4 > 0
            if j1_valid and j1 < ys:
                ys = j1
            if j2_valid and j2 < ys:
                ys = j2
            if j1_valid and j1 > ye:
                ye = j1
            if j2_valid and j2 > ye:
                ye = j2

    return xs, xe, ys, ye


def center_resize_digit_in_image(digit_img, padding):
    # Slice the digit
    # [x, y, w, h] = cv.boundingRect(box)
    # digit_box = digit_img[x:x+h, y:y+w]
    xs, xe, ys, ye = find_number_bound_enhanced(digit_img)
    h = xe - xs
    w = ye - ys
    digit_box = digit_img[xs:xe, ys:ye]

    # print(f'digit image shape: {digit_img.shape}, Rect:{(x, y), (x+h, y), (x, y+w), (x+h, y+w)}')
    # print(f'bounding digit h:{h}, w:{w}')
    # print(f'digit box shape: {digit_box.shape}')

    # Show Bounding Box
    # dd = cv.cvtColor(digit_image, cv.COLOR_GRAY2BGR)
    # dd[x, y] = (0, 0, 255); dd[x+h, y] = (0, 0, 255); dd[x, y+w] = (0, 0, 255); dd[x+h, y+w] = (0, 0, 255)
    # show('bounding digit', dd, False)


    # show('digit box', digit_box, False)
    # show('digit image', digit_img)
    # cv.destroyWindow('digit box')

    # convert sliced digit image to square
    if h > w:
        pad = (h - w) // 2
        M = np.float32([
            [1, 0, pad],
            [0, 1, 0]
        ])
        digit_box = cv.warpAffine(digit_box, M, (h, h))
    elif w > h:
        pad = (w - h) // 2
        M = np.float32([
            [1, 0, 0],
            [0, 1, pad]
        ])
        digit_box = cv.warpAffine(digit_box, M, (w, w))
    else:  # w == h
        pass

    # Resize digit image to digit_size with padding
    digit_scaled_side = 32 - (2 * padding)
    digit_box_scaled = cv.resize(digit_box, (digit_scaled_side, digit_scaled_side))
    normalized_digit_image = np.zeros((32, 32), dtype=np.uint8)
    normalized_digit_image[padding:padding + digit_scaled_side, padding:padding + digit_scaled_side] = digit_box_scaled

    # DEBUGGING
    # show('padded digit', digit_box)
    # show('1', digit_box_scaled)
    # show('1', normalized_digit_image)
    # cv.waitKey(0)

    return normalized_digit_image


# Rule is: 32 - 8 % 8 == 0
winSize = (32, 32)
blockSize = (8, 8)
blockStride = (8, 8)
cellSize = (4, 4)
nbins = 9
hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
def guess_digit_knn(digit_image, model, min_contour_area):
    # Check if cell is empty
    contours, _ = cv.findContours(digit_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    try:
        [x, y, w, h] = cv.boundingRect(contours[0])
        flag = w > 6 and h > 10
    except:
        flag = False

    # if len(contours) > 0 and cv.contourArea(contours[0]) > min_contour_area and w > 10 and h > 10:
    if flag:
        # Remove other small contours:
        if len(contours) > 1:
            mask = np.ones(digit_image.shape[:2], dtype=np.uint8) * 255
            # f = digit_image.copy()
            # f = cv.cvtColor(f, cv.COLOR_GRAY2BGR)
            cv.drawContours(mask, contours[1:], -1, color=(0, 0, 255), thickness=3)
            digit_image = cv.bitwise_and(digit_image, digit_image, mask=mask)
            # cv.imshow('mask', mask)
            # cv.imshow('digit image', digit_image)
            # cv.waitKey(0)
            # cv.destroyWindow('contours')

        # Normalize digit image
        centerized_resized_digit_image = center_resize_digit_in_image(digit_image, padding=2)

        # Extract features from image
        digit_feature = hog.compute(centerized_resized_digit_image, None, None, locations=[])

        # Predict result
        ret, result, neighbours, dist = model.findNearest(np.array([digit_feature]), k=5)
        digit_number = [int(result[0][0])]

        # [DEBUG] print(digit_number)
        # [DEBUG] show('digit', centerized_resized_digit_image)
    else:
        # Contour is empty or too small => empty cell
        digit_number = []
    # print('cell>', digit_number)
    # show('[DEBUG]', digit_image)
    # cv.destroyWindow('digit box')
    return digit_number

def guess_digit_svm(digit_image, model):
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

    # Check if cell is empty
    contours, _ = cv.findContours(digit_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    try:
        [x, y, w, h] = cv.boundingRect(contours[0])
        flag = w > 6 and h > 10
    except:
        flag = False

    # if len(contours) > 0 and cv.contourArea(contours[0]) > min_contour_area and w > 10 and h > 10:
    if flag:
        # Remove other small contours:
        if len(contours) > 1:
            mask = np.ones(digit_image.shape[:2], dtype=np.uint8) * 255
            # f = digit_image.copy()
            # f = cv.cvtColor(f, cv.COLOR_GRAY2BGR)
            cv.drawContours(mask, contours[1:], -1, color=(0, 0, 255), thickness=3)
            digit_image = cv.bitwise_and(digit_image, digit_image, mask=mask)
            # cv.imshow('mask', mask)
            # cv.imshow('digit image', digit_image)
            # cv.waitKey(0)
            # cv.destroyWindow('contours')

        # Normalize digit image
        centerized_resized_digit_image = center_resize_digit_in_image(digit_image, padding=2)

        # Extract features from image
        digit_feature = hog(deskew(centerized_resized_digit_image)).reshape(1, 64).astype(np.float32)

        # Predict result
        ret, result = model.predict(np.array(digit_feature))
        digit_number = [int(result[0][0])]
    else:
        # Contour is empty or too small => empty cell
        digit_number = []
    # print('cell>', digit_number)
    # show('[DEBUG]', digit_image)
    # cv.destroyWindow('digit box')
    return digit_number

def load_templates():
    numbers = []
    for i in range(1, 10):
        num = cv.imread(f"Assets/Templates/{i}.jpg", cv.IMREAD_GRAYSCALE)
        num = cv.resize(num, (24, 32), interpolation=cv.INTER_NEAREST)
        _, num = cv.threshold(num, 50, 255, cv.THRESH_BINARY_INV)
        padded = np.hstack((np.zeros((32, 4)), num))
        padded = np.hstack((padded, np.zeros((32, 4))))
        # show('p', padded)
        numbers.append(padded)
    numbers = np.array(numbers)
    return numbers

templates = load_templates()
def guess_digit_intersect(digit_image):
    # Check if cell is empty
    contours, _ = cv.findContours(digit_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    try:
        [x, y, w, h] = cv.boundingRect(contours[0])
        flag = w > 6 and h > 10
    except:
        flag = False

    # if len(contours) > 0 and cv.contourArea(contours[0]) > min_contour_area and w > 10 and h > 10:
    if flag:
        # Remove other small contours:
        if len(contours) > 1:
            mask = np.ones(digit_image.shape[:2], dtype=np.uint8) * 255
            # f = digit_image.copy()
            # f = cv.cvtColor(f, cv.COLOR_GRAY2BGR)
            cv.drawContours(mask, contours[1:], -1, color=(0, 0, 255), thickness=3)
            digit_image = cv.bitwise_and(digit_image, digit_image, mask=mask)
            # cv.imshow('mask', mask)
            # cv.imshow('digit image', digit_image)
            # cv.waitKey(0)
            # cv.destroyWindow('contours')

        # Normalize digit image
        centerized_resized_digit_image = center_resize_digit_in_image(digit_image, padding=2)
        show('a', centerized_resized_digit_image)
        # centerized_resized_digit_image = cv.threshold(centerized_resized_digit_image, 200, 1, type=cv.THRESH_BINARY)
        # show('a', centerized_resized_digit_image)

        # Predict result
        matches = []
        for template in templates:
            apply_template = cv.bitwise_and(template.astype(np.float32), centerized_resized_digit_image.astype(np.float32))
            match_percent = sum(map(sum, apply_template // 255))
            # print(match_percent)
            # show('t', apply_template)
            # match_percent = np.sum(apply_template//255) / 32*32
            # print(match_percent)
            matches.append(match_percent)

        digit_number = [np.argmax(matches) + 1]
        print('-------------------\n' + str(digit_number[0]))
        show('digit', centerized_resized_digit_image)
    else:
        # Contour is empty or too small => empty cell
        digit_number = []
    # print('cell>', digit_number)
    # show('[DEBUG]', digit_image)
    # cv.destroyWindow('digit box')
    return digit_number

def extract_digits(sudoko_board, model):
    decoded_digits = []
    digits = []
    rows = np.vsplit(sudoko_board, 9)
    for row in rows:
        decoded_row_digits = []
        row_digits = []
        cols = np.hsplit(row, 9)
        numbers_in_row_count = 0
        for digit_image in cols:
            digit = guess_digit_knn(digit_image, model, min_contour_area=10)
            # digit = guess_digit_svm(digit_image, model)
            # digit = guess_digit_intersect(digit_image)
            if len(digit) != 0:  # if number found check flag
                numbers_in_row_count = numbers_in_row_count + 1
            row_digits.append(digit)
            decoded_row_digits.append(0 if len(digit) == 0 else digit[0])
        # Performance check
        if numbers_in_row_count < 1:
            raise InterruptedError
        digits.append(row_digits)
        decoded_digits.append(decoded_row_digits)
    return decoded_digits, digits


def project_solution_on_frame(resultion, solved_sudoko, digits_list, corners, frame, numbers):
    def generate_solved_square_image(solved_sudoko, digits_list, numbers):
        image = np.zeros((288, 288), dtype=np.uint8)
        for i in range(9):
            for j in range(9):
                if len(digits_list[i][j]) == 0:
                    key = solved_sudoko[i][j][0]
                else:
                    key = 0

                num_img = numbers[key]
                image[i*32:i*32+32, j*32:j*32+32] = num_img
        return image

    def warp_board(corners, solved_image, dimension):
        top_left, top_right, bottom_right, bottom_left = corners[0], corners[1], corners[2], corners[3]
        side = 288
        src = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
        dst = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
        m = cv.getPerspectiveTransform(src, dst)
        return cv.warpPerspective(solved_image, m, dimension)

    # Generate square image with digits_list (288x288)
    image = generate_solved_square_image(solved_sudoko, digits_list, numbers)

    # Warp solved image to original frame
    warped_board = warp_board(corners, image, resultion)

    # Add image to orginal image
    # img1 = cv.cvtColor(gray_frame, cv.COLOR_GRAY2BGR)
    img1 = frame
    img2 = np.zeros((resultion[1], resultion[0], 3), dtype=np.uint8)
    img2[:, :, 2] = warped_board
    img2_gray = warped_board
    mask = warped_board
    mask_inv = cv.bitwise_not(warped_board)
    img1_bg = cv.bitwise_and(img1, img1, mask=mask_inv)
    img2_fg = img2
    final_img = cv.add(img1_bg, img2_fg)
    return final_img

