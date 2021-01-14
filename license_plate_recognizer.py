# test file if you want to quickly try tesseract on a license plate image
from os.path import join

import cv2
import easyocr
from collections import Counter
from draw import draw_detect, put_text_pil
from plate_image import preprocess_image, parse_result, deskew
from process_contour import crop_plate

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

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
width_coef = 1280 / im_w
height_coef = 720 / im_h
# allow_symbols = 'УКЕНХВАРОСМТ1234567890'
allow_symbols = 'ETYOPAHKXCBM1234567890'

res_arr = list()
true_numbers = list()
while rr:
    frame_res = None
    ret, frame = videofile.read()
    if not ret:
        break

    bboxes = model.predict_img(frame, plot_img=False, show_text=False)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for bbox in bboxes.values:
        height_coef = 80 / bbox[7]
        width_coef = 400 / bbox[6]

        crop_img = frame[int(bbox[1] - 7):int(bbox[1]) + int(bbox[7] + 7),
                   int(bbox[0] - 7):int(bbox[0]) + int(bbox[6] + 7)]

        crop_img = preprocess_image(crop_img, width_coef=width_coef, height_coef=height_coef, show_steps=False)
        crop_img = crop_plate(crop_img)
        crop_img = deskew(crop_img)

        pred = reader.detect(crop_img)
        # result = reader.recognize(crop_img, pred[0], pred[1], allowlist=allow_symbols, detail=0)
        # print('detection: {}'.format(result))

        tess_res = pytesseract.image_to_string(crop_img, lang='eng')
        # print('tess res: {}'.format(tess_res))

        result = parse_result(reader.readtext(crop_img, allowlist=allow_symbols, detail=0, adjust_contrast=0.5))

        if len(result) >= 6:
            res_arr.append(result)


        print('read text: {}'.format(result))
        # cv2.imshow('crop_img', crop_img)
        # cv2.waitKey()
        c = Counter(res_arr)
        for key, values in c.items():
            if values >= 10:
                print("___________________\n nomer: {}".format(key))
                true_numbers.append(key)
                # del c[items]
                # for el in res_arr:
                #     if el == key:
                #         del el
                frame_res = draw_detect(frame, bboxes)
                # frame_res = put_text_pil(frame_res, key, (bbox[0], bbox[1]))
                cv2.putText(frame_res, key, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", 1000, 600)
    if frame_res is not None:
        cv2.imshow('output', frame_res)
    else:
        cv2.imshow('output', frame)
    if cv2.waitKey(1) & 0xff == ord('&'):
        break

    # cv2.waitKey(1)

#
# output_video.release()
cv2.destroyAllWindows()
