import operator
import cv2 as cv
import ModelsUtil
import numpy as np
from Solver.fast_solve import solve, print_board
from Validator import isValidSudoku, NotValidBoard
from Utils import show, equalize_image, normalize_sudoko_board, extract_digits, project_solution_on_frame


def load_numbers():
    numbers = dict()
    numbers[0] = np.zeros((32, 32))
    for i in range(1, 10):
        num = cv.imread(f"Assets/Projection Numbers/{i}.jpg", cv.IMREAD_GRAYSCALE)
        num = cv.resize(num, (24, 32), interpolation=cv.INTER_NEAREST)
        _, num = cv.threshold(num, 50, 255, cv.THRESH_BINARY_INV)
        padded = np.hstack((np.zeros((32, 4)), num))
        padded = np.hstack((padded, np.zeros((32, 4))))
        numbers[i] = padded
    return numbers


def pre_process_gray_image(grayed_frame, dilate=True):
    """Apply Gaussian blur, Adaptive thresholding and dilation with Cross filter"""
    proc = cv.GaussianBlur(grayed_frame, (9, 9), 0)
    proc = cv.adaptiveThreshold(proc, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    kernel = np.array([[0., 1., 0.],
                       [1., 1., 1.],
                       [0., 1., 0.]], np.uint8)
    proc = cv.dilate(proc, kernel)
    return proc


def find_corners_of_largest_polygon(processed_img):
    """Finds the 4 corners of the largest contour area in the image"""
    contours, h = cv.findContours(processed_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    polygon = contours[0]

    # Draw contour: You need to pass image frame copy to here to show contours:
    # f = frame.copy()
    # cv.drawContours(f, polygon, -1, color=(0, 0, 255), thickness=2)
    # cv.imshow('contours', f)
    # cv.waitKey(0)
    # cv.destroyWindow('contours')

    # Bottom-right point has the largest (x + y) value
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    # Top-left has point smallest (x + y) value
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    # The bottom-left point has the smallest (x — y) value
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    # The top-right point has the largest (x — y) value
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


def crop_warp_resize_frame(frame, corners):
    """Crops and warps a square section from the frame according to the provided corners"""
    top_left, top_right, bottom_right, bottom_left = corners[0], corners[1], corners[2], corners[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    side = max([distance_between(bottom_right, top_right),
                distance_between(top_left, bottom_left),
                distance_between(bottom_right, bottom_left),
                distance_between(top_left, top_right)])
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    m = cv.getPerspectiveTransform(src, dst)
    warped_frame = cv.warpPerspective(frame, m, (int(side), int(side)))
    return cv.resize(warped_frame, (288, 288))  # each digit is 32x32 pixels


def distance_between(p1, p2):
    """Returns the distance between two points"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def extract_grid(digits_images, numbers_features):
    digits_numbers = []
    for digit_img in digits_images:
        probs = []
        for num_feature in numbers_features:
            print(digit_img.shape)
            print(num_feature.shape)
            result = np.tensordot(digit_img, num_feature, axes=((0, 1), (0, 1)))
            probs.append(int(result))
        print(probs)
        max_prob_index = np.where(probs == np.max(probs))
        digit_number = max_prob_index[0] + 1
        digits_numbers.append(digit_number)
        print(digit_number)
        cv.waitKey(0)


if __name__ == '__main__':
    # Load numbers for projection (32x32)
    numbers_dict = load_numbers()

    # Init trained Recogntion Model
    model = ModelsUtil.get_trained_knn()
    # modle = ModelsUtil.get_trained_svm()

    vid = cv.VideoCapture(0, cv.CAP_DSHOW)
    resulution = (640, 480)
    vid.set(cv.CAP_PROP_FRAME_WIDTH, resulution[0])
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, resulution[1])
    # vid.open("http://192.168.1.100:8080/video")
    print(f'CAMERA RES ({int(vid.get(cv.CAP_PROP_FRAME_WIDTH))}x{int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))})')

    # [skip frames] i = 0
    frame_title = 'CAMERA'
    pause = False
    solved = False
    solved_frame_count = 0
    solved_frame_limit = 5
    end_frame = np.zeros((288, 288), dtype=np.uint8)
    solved_sudoko = []
    sudoko_digits = []
    while True:
        # Get user input
        k = cv.waitKey(5)
        if k == ord(' '):  # Space for closing app
            break
        elif k == ord('p'):  # P for pause frame
            pause = not pause
            continue

        if pause:
            continue

        # Capture the frame
        _, frame = vid.read()

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # [DEBUG] show('frame', gray_frame)

        # PreProcess the image (Blur, Thresholding, Dilate)
        proc_frame = pre_process_gray_image(gray_frame)
        # [DEBUG] show('pre-processed frame', proc_frame)

        # Find corners of largest polygon (Get largest polygon)
        corners = find_corners_of_largest_polygon(proc_frame)

        # Crop & Wrap image => Then Resize it to (288x288)
        square_gray_frame = crop_warp_resize_frame(gray_frame, corners)

        # Equalize gray image for better performance:
        square_gray_frame = equalize_image(square_gray_frame, 'CLAHE')
        show('cropped warped re-sized gray frame', square_gray_frame, False)

        # Extract Sudoku numbers (Remove grid lines , Blur(Gaussian OR Bilateral),Threshold,Opening(2, 2),Closing(3, 3)
        sudoku_board = normalize_sudoko_board(square_gray_frame, fast=True)
        show('normalized board', sudoku_board, False)

        try:
            if solved:
                solved_frame_count = solved_frame_count + 1
                if solved_frame_count == solved_frame_limit:
                    solved = False
                cv.setWindowTitle(frame_title, 'CAMERA - SOLVED')
                end_frame = project_solution_on_frame(resulution, solved_sudoko, sudoko_digits, corners, frame,
                                                      numbers_dict)
            else:
                decoded_sudoko_digits, sudoko_digits = extract_digits(sudoku_board, model)
                # print('Extracted Sudoko:')
                # print_board(sudoko_digits)
                # print('--------------------------------------')
                if not isValidSudoku(decoded_sudoko_digits):
                    raise NotValidBoard
                solved_sudoko = solve(sudoko_digits)
                solved = True
                solved_frame_count = 1
                # print('Solved Sudoko:')
                # print_board(solved_sudoko)
                cv.setWindowTitle(frame_title, 'CAMERA - SOLVED')
                end_frame = project_solution_on_frame(resulution, solved_sudoko, sudoko_digits, corners, frame, numbers_dict)
        except (TypeError, InterruptedError, NotValidBoard, Exception):
            # print('Invalid Sudoku board!!')
            cv.setWindowTitle(frame_title, 'CAMERA - INVALID SUDOKU BOARD')
            end_frame = frame
        cv.imshow(frame_title, end_frame)

    # Camera Closed
    vid.release()
    cv.destroyAllWindows()
