from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2
import os
import time
import matplotlib.pyplot as plt

config_file = "../configs/my_test_e2e_mask_rcnn_R_50_FPN_1x.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE","cuda"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

#load image and then run prediction
# image = cv2.imread('001.png')
# predictions ,bbox= coco_demo.run_on_opencv_image(image)
#
# cv2.namedWindow('predictionWindow',cv2.WINDOW_NORMAL)
# cv2.imshow('predictionWindow',predictions)
# cv2.waitKey(0)

timeSum=0
cnt=0

for imageName in os.listdir('../datasets/coco/val2017'):
    image=cv2.imread('../datasets/coco/val2017/'+imageName)
    start = time.perf_counter()
    predictions, _, _, _ = coco_demo.run_on_opencv_image(image)
    stop = time.perf_counter()
    timeInterval = stop - start
    timeSum += timeInterval
    cnt+=1
    #cv2.imwrite('valResults/'+imageName,predictions)

print("模板匹配时间："+str(timeSum / cnt) + "秒")
