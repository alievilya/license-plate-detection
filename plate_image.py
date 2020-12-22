# -*- coding: utf_8 -*-
import pytesseract
import cv2
import os
import numpy as np
from os.path import join
from operator import itemgetter
from draw import draw_detect
import easyocr
import scipy.fftpack # For FFT2
from test_contour import find_biggest_rect, crop_plate
#### imclearborder definition

def imclearborder(imgBW, radius):

    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    try:
        contours, hierarchy = cv2.findContours(imgBWcopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(imgBWcopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]

    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(imgBWcopy, contours, -1, 255, 3)

        # find the biggest countour (c) by the area
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # draw the biggest contour (c) in green
        cv2.rectangle(imgBWcopy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('rectangles', imgBWcopy)
        cv2.waitKey(0)

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy

#### bwareaopen definition
def bwareaopen(imgBW, areaPixels):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    _, contours, _ = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels ):
            cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy


def delete_borders(crop_img):
    # Number of rows and columns
    rows = crop_img.shape[0]
    cols = crop_img.shape[1]

    # Convert image to 0 to 1, then do log(1 + I)
    imgLog = np.log1p(np.array(crop_img, dtype="float") / 255)

    # Create Gaussian mask of sigma = 10
    M = 2 * rows + 1
    N = 2 * cols + 1
    sigma = 10
    (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
    centerX = np.ceil(N / 2)
    centerY = np.ceil(M / 2)
    gaussianNumerator = (X - centerX) ** 2 + (Y - centerY) ** 2

    # Low pass and high pass filters
    Hlow = np.exp(-gaussianNumerator / (2 * sigma * sigma))
    Hhigh = 1 - Hlow

    # Move origin of filters so that it's at the top left corner to
    # match with the input image
    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

    # Filter the image and crop
    If = scipy.fftpack.fft2(imgLog.copy(), (M, N))
    Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M, N)))
    Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M, N)))

    # Set scaling factors and add
    gamma1 = 0.3
    gamma2 = 1.5
    Iout = gamma1 * Ioutlow[0:rows, 0:cols] + gamma2 * Iouthigh[0:rows, 0:cols]

    # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255 * Ihmf, dtype="uint8")

    # Threshold the image - Anything below intensity 65 gets set to white
    Ithresh = Ihmf2 < 65
    Ithresh = 255 * Ithresh.astype("uint8")

    # Clear off the border.  Choose a border radius of 5 pixels
    Iclear = imclearborder(Ithresh, 1)

    # Eliminate regions that have areas below 120 pixels
    Iopen = bwareaopen(Iclear, 120)

    # Show all images
    cv2.imshow('Original Image', crop_img)
    cv2.imshow('Homomorphic Filtered Result', Ihmf2)
    cv2.imshow('Thresholded Result', Ithresh)
    cv2.imshow('Opened Result', Iopen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return Iopen


def parse_result(res):
    not_allowed_syms = (' ', '!', '$', '%', '^', '&', '*', '(', ')', '+', '=', '_', '-', '[', ']',
                        'q', 'w', 'r', 'u', 'i', 's', 'd', 'f', 'g', 'j', 'l', 'z', 'v', 'n', '<', '>')
    nums = ('1234567890')
    B_similar = ['8', '3']
    new_el = ""
    for el in res:
        if len(el) <= 5: continue
        for i, sym in enumerate(el):
            if sym in not_allowed_syms:
                new_el += ''
            else: new_el += sym

            if i == 0 and sym in B_similar:
                new_el += 'B'

    return new_el


def preprocess_image(img, width_coef, height_coef, show_steps=False):
    # gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # equ = cv2.equalizeHist(crop_img)
    # res = np.hstack((crop_img, equ))  # stacking images side-by-side

    # gray = cv2.resize(img, None, fx=width_coef, fy=height_coef, interpolation=cv2.INTER_AREA)
    disnoised = cv2.fastNlMeansDenoising(img, None, 5, 5, 7)

    blur = cv2.GaussianBlur(disnoised, (5, 5), 0)
    gray = cv2.medianBlur(blur, 3)
    # se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # inversed_im = cv2.bitwise_not(thresh)

    if show_steps:
        cv2.imshow('input', img)
        cv2.imshow('disnoised', disnoised)
        cv2.imshow('blur', gray)


        # cv2.imshow("Otsu", thresh)
        # cv2.imshow("Result", cropped)
        # cv2.imshow("bitwise_not", inversed_im)
        cv2.waitKey(0)
    return gray

if "__name__" == "__main__":

    from models import Yolov4

    model = Yolov4(weight_path='custom.weights',
                   class_name_path='class_names/plate.txt')

    reader = easyocr.Reader(['en', 'ru'])

    for filename in os.listdir('data_files/images'):

    # filename = 'data_files/test4.jpg'
        filepath = os.path.join('data_files/images', filename)
        image = cv2.imread(filepath, 0)
        bboxes = model.predict(filepath, plot_img=False)
        allow_symbols = 'УЕОРАВСМТКETOPAHKXCBM1234567890'

        # pred = reader.detect(image)
        # res = reader.recognize(image, pred[0], pred[1], allowlist=allow_symbols, detail=1)
        # # res.sort(key=lambda x: x[0][0][0])
        # # print(res)


        for bbox in bboxes.values:
            height_coef = 80 / bbox[7]
            width_coef = 350 / bbox[6]

            crop_img = image[int(bbox[1]-10):int(bbox[1]) + int(bbox[7]+10), int(bbox[0]-10):int(bbox[0]) + int(bbox[6]+10)]

            # crop_img = preprocess_image(crop_img, width_coef=width_coef, height_coef=height_coef, show_steps=False)
            crop_img = crop_plate(crop_img)
            pred = reader.detect(crop_img)
            result = reader.recognize(crop_img, pred[0], pred[1], allowlist=allow_symbols, detail=0)
            print('detection: {}'.format(result))

            result = parse_result(reader.readtext(crop_img, allowlist=allow_symbols, detail=0))
            print('read text: {}'.format(result))



        frame_res = draw_detect(image, bboxes)
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("output", 1000, 600)
        cv2.imshow('output', frame_res)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
