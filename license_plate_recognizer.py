# test file if you want to quickly try tesseract on a license plate image
import pytesseract
import cv2
import os
import numpy as np
from os.path import join
from draw import draw_detect
import easyocr
from test_contour import crop_plate
from plate_image import preprocess_image, parse_result



show_steps = False


from models import Yolov4
model = Yolov4(weight_path='custom.weights',
               class_name_path='class_names/plate.txt')
reader = easyocr.Reader(['en', 'ru'])


# bboxes = model.predict('test2.jpg')
# image = cv2.imread("test2.jpg", 0)

output_format = 'avi'
video_name = 'test2.mp4'
video_path = join('data_files', video_name)
output_name = 'out_' + video_name[0:-3] + output_format

videofile = cv2.VideoCapture(video_path)
fps = videofile.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output_video = cv2.VideoWriter(output_name, fourcc, fps, (640, 480))
rr, first_frame = videofile.read()
im_w = first_frame.shape[1]
im_h = first_frame.shape[0]
width_coef = 1280/im_w
height_coef = 720/im_h
allow_symbols = 'УЕОРАВСМТКETOPAHKXCBM1234567890'

while rr:
    ret, frame = videofile.read()
    if not ret:
        break

    bboxes = model.predict_img(frame, plot_img=False, show_text=False)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # pred = reader.detect(image)
    # res = reader.recognize(image, pred[0], pred[1], allowlist=allow_symbols, detail=1)
    # # res.sort(key=lambda x: x[0][0][0])
    # # print(res)

    for bbox in bboxes.values:
        height_coef = 80 / bbox[7]
        width_coef = 350 / bbox[6]

        crop_img = frame[int(bbox[1] - 10):int(bbox[1]) + int(bbox[7] + 10),
                   int(bbox[0] - 10):int(bbox[0]) + int(bbox[6] + 10)]

        # crop_img = preprocess_image(crop_img, width_coef=width_coef, height_coef=height_coef, show_steps=False)
        crop_img = crop_plate(crop_img)
        pred = reader.detect(crop_img)
        result = reader.recognize(crop_img, pred[0], pred[1], allowlist=allow_symbols, detail=0)
        print('detection: {}'.format(result))

        result = parse_result(reader.readtext(crop_img, allowlist=allow_symbols, detail=0))
        print('read text: {}'.format(result))

    frame_res = draw_detect(frame, bboxes)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", 1000, 600)
    cv2.imshow('output', frame_res)
    if cv2.waitKey(1) & 0xff == ord('&'):
        break


    # cv2.waitKey(1)

#
# output_video.release()
cv2.destroyAllWindows()