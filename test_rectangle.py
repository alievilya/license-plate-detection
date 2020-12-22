# res = [([[726, 330], [758, 330], [758, 362], [726, 362]], '05', 0.9947222471237183),
#        ([[687, 341], [729, 341], [729, 381], [687, 381]], 'HМ', 0.9953700304031372), (
#            [[628.6263943605131, 341.1923802617959], [693.82987126915, 336.78564375624916],
#             [696.3736056394869, 385.8076197382041], [630.17012873085, 390.21435624375084]], '1О0', 0.3260660767555237),
#        ([[605, 355], [633, 355], [633, 393], [605, 393]], 'K', 0.541047215461731),
#        ([[727, 359], [749, 359], [749, 373], [727, 373]], 'BO5', 0.16357183456420898)]

# ress = [([[698, 288], [724, 288], [724, 316], [698, 316]], '62', 0.8477792143821716), (
# [[604.5611569566013, 309.8295040435816], [702.3647139015062, 289.836969634746], [709.4388430433987, 329.1704959564184],
#  [611.6352860984938, 348.163030365254]], 'P6790K', 0.7591648697853088)]

# Specify device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

# Import all necessary libraries.
import numpy as np
import sys
import matplotlib.image as mpimg

# NomeroffNet path
NOMEROFF_NET_DIR = os.path.abspath('X:/code/rabota/nomeroff-net-master/')
sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import  Detector
from NomeroffNet import  filters
from NomeroffNet import  RectDetector
from NomeroffNet import  OptionsDetector
from NomeroffNet import  TextDetector
from NomeroffNet import  textPostprocessing

# load models
rectDetector = RectDetector()

optionsDetector = OptionsDetector()
optionsDetector.load("latest")

textDetector = TextDetector.get_static_module("eu")()
textDetector.load("latest")

nnet = Detector()
nnet.loadModel(NOMEROFF_NET_DIR)

# Detect numberplate
img_path = 'images/example2.jpeg'
img = mpimg.imread(img_path)

# Generate image mask.
cv_imgs_masks = nnet.detect_mask([img])

for cv_img_masks in cv_imgs_masks:
    # Detect points.
    arrPoints = rectDetector.detect(cv_img_masks)

    # cut zones
    zones = rectDetector.get_cv_zonesBGR(img, arrPoints, 64, 295)

    # find standart
    regionIds, stateIds, countLines = optionsDetector.predict(zones)
    regionNames = optionsDetector.getRegionLabels(regionIds)

    # find text with postprocessing by standart
    textArr = textDetector.predict(zones)
    textArr = textPostprocessing(textArr, regionNames)
    print(textArr)
    # ['JJF509', 'RP70012']