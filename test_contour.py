import cv2
import numpy as np
import imutils
from skimage import exposure

def clahe(img, clip_limit=2.0, grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)

def find_biggest_rect(src):



    im, contours, hierarchy = cv2.findContours(src,
                                               cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)

    # Filter the rectangle by choosing only the big ones
    # and choose the brightest rectangle as the bed
    max_brightness = 0
    brightest_rectangle = [0, 0, 0, 0]
    canvas = src.copy()
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        if w*h > 4000:
            mask = np.zeros(src.shape, np.uint8)
            mask[y:y+h, x:x+w] = src[y:y+h, x:x+w]
            brightness = np.sum(mask)
            if brightness > max_brightness:
                brightest_rectangle = rect
                max_brightness = brightness
            # cv2.imshow("mask", mask)
            # cv2.waitKey(0)

    x, y, w, h = brightest_rectangle
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 1)


    return canvas

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# load the image and grab the source coordinates (i.e. the list of
# of (x, y) points)
# NOTE: using the 'eval' function is bad form, but for this example
# let's just roll with it -- in future posts I'll show you how to
# automatically determine the coordinates without pre-supplying them
def crop_plate(image):
    img = image.copy()
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, 255, 3)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    img = img[y:y + h, x:x + w]
    return img

if "__name__" == "__main__":
    image = cv2.imread('data_files/xx.jpg', 0)

    blur = cv2.GaussianBlur(image, (5, 5), 0)
    gray = cv2.medianBlur(blur, 3)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, 255, 3)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Result", image[y:y + h, x:x + w])

    cv2.waitKey(0)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    # Going through every contours found in the image.
    # for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
    # draws boundary of contours.
    cv2.drawContours(image, [approx], 0, (0, 0, 255), 5)
    # Used to flatted the array containing
    # the co-ordinates of the vertices.
    n = approx.ravel()
    i = 0
    lists = []
    arr = []
    for j in n:
        if (i % 2 == 0):
            x = n[i]
            y = n[i + 1]
            # String containing the co-ordinates.
            string = str(x) + " " + str(y)
            if (i == 0):
                # text on topmost co-ordinate.
                cv2.putText(image, "Arrow tip", (x, y),
                            cv2.FONT_HERSHEY_COMPLEX , 0.5, (255, 0, 0))
            else:
                # text on remaining co-ordinates.
                cv2.putText(image, string, (x, y),
                            cv2.FONT_HERSHEY_COMPLEX , 0.5, (0, 255, 0))
            if len(arr)> 4:
                continue
            else:
                arr.append((x, y))
        i = i + 1

    # Showing the final image.
    cv2.imshow('image2', image)

    pts = np.array(np.array(arr), dtype = "float32")

    # apply the four point tranform to obtain a "birds eye view" of
    # the image


    warped = four_point_transform(image, pts)
    # show the original and warped images
    #
    cv2.imshow("Original", image)
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)