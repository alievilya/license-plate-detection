# test file if you want to quickly try tesseract on a license plate image
import pytesseract
import cv2
import os
import numpy as np
from os.path import join
from draw import draw_detect
import easyocr
from test_contour import crop_plate

def parse_result(res):
    not_allowed_syms = [' ', '!', '$', '%', '^', '&', '*', '(', ')', '+', '=', '_', '-', '[', ']']
    new_el = ""
    for el in res:
        if len(el) <= 5: continue
        for i, sym in enumerate(el):
            if sym in not_allowed_syms:
                new_el += ''
            else: new_el += sym
    return new_el

show_steps = False



from models import Yolov4
model = Yolov4(weight_path='custom.weights',
               class_name_path='class_names/plate.txt')
reader = easyocr.Reader(['en'])


# bboxes = model.predict('test2.jpg')
# image = cv2.imread("test2.jpg", 0)

output_format = 'avi'
video_name = 'test3.mp4'
video_path = join('data_files', video_name)
output_name = 'out_' + video_name[0:-3] + output_format

videofile = cv2.VideoCapture(video_path)
fps = videofile.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output_video = cv2.VideoWriter(output_name, fourcc, fps, (640, 480))
rr, first_frame = videofile.read()

while rr:
    ret, frame = videofile.read()
    if not ret:
        break

    bboxes = model.predict_img(frame, plot_img=False, show_text=False)

    for bbox in bboxes.values:

        crop_img = frame[int(bbox[1] - 5):int(bbox[1]) + int(bbox[7] + 10),
                   int(bbox[0] - 5):int(bbox[0]) + int(bbox[6] + 10)]

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        # equ = cv2.equalizeHist(crop_img)
        # res = np.hstack((crop_img, equ))  # stacking images side-by-side

        gray = cv2.resize(gray, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
        disnoised = cv2.fastNlMeansDenoising(gray, None, 10, 10, 7)

        blur = cv2.GaussianBlur(disnoised, (5, 5), 0)
        gray = cv2.medianBlur(blur, 3)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        cropped_plate = crop_plate(thresh)
        inversed_im = cv2.bitwise_not(thresh)

        if show_steps:
            cv2.imshow('blur', gray)
            cv2.imshow('crop', crop_img)
            cv2.imshow('disnoised', disnoised)
            cv2.imshow("Otsu", thresh)
            cv2.imshow("Result", cropped_plate)
            cv2.imshow("bitwise_not", inversed_im)

            # cv2.imshow("border_im", border_im)
            cv2.waitKey(0)

        # del_b = delete_borders(thresh)
        # im2 = gray.copy()
        im2 = thresh.copy()

        # need to run only once to load model into memory
        # result = parse_result(reader.readtext(im2, detail=0))
        # print('default: {}'.format(result))
        #
        # result = parse_result(reader.readtext(inversed_im, detail=0))
        # print('inversed: {}'.format(result))

        result = parse_result(reader.readtext(cropped_plate, detail=0))
        print('with cropped plate: {}'.format(result))

        # plate_num = ""
        # # loop through contours and find letters in license plate
        # for cnt in sorted_contours:
        #     x,y,w,h = cv2.boundingRect(cnt)
        #     height, width = im2.shape
        #
        #     # if height of box is not a quarter of total height then skip
        #     if height / float(h) > 6: continue
        #     ratio = h / float(w)
        #     # if height to width ratio is less than 1.5 skip
        #     if ratio < 1.5: continue
        #     area = h * w
        #     # if width is not more than 25 pixels skip
        #     if width / float(w) > 15: continue
        #     # if area is less than 100 pixels skip
        #     if area < 100: continue
        #     # draw the rectangle
        #     rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
        #     roi = thresh[y-5:y+h+5, x-5:x+w+5]
        #     roi = cv2.bitwise_not(roi)
        #     roi = cv2.medianBlur(roi, 5)
        #     #cv2.imshow("ROI", roi)
        #     #cv2.waitKey(0)
        #     text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
        #     #print(text)
        #     plate_num += text[0]
        #
        # print(plate_num)
        # cv2.imshow("Character's Segmented", im2)

    frame_res = draw_detect(frame, bboxes)
    # cv2.imshow('d0', frame_res)
    cv2.waitKey(1)

#
# output_video.release()
cv2.destroyAllWindows()