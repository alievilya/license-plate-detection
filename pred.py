from models import Yolov4
model = Yolov4(weight_path='custom.weights',
               class_name_path='class_names/plate.txt')
model.predict('test2.jpg')