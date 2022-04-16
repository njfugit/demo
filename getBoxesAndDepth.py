import cv2
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt

from templateMatching import TemplateMatching
from mouth import Mouth
from predictor import COCODemo
from maskrcnn_benchmark.config import cfg



class GetBoxesAndDepth(object):

    def __init__(self, detect_target='mouth'):
        #print("Begining to initialize Detect4CCode instance...")
        config_file = "../configs/my_test_e2e_mask_rcnn_R_50_FPN_1x.yaml"

        # update the config options with the config file
        cfg.merge_from_file(config_file)

        # manual override some options
        cfg.merge_from_list(["MODEL.DEVICE", "cuda"])###cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

        self.mouth=Mouth(usingNetwork=True)
        self.coco_demo_tonsil = COCODemo(
            cfg,
            min_image_size=800,#400,#800,
            confidence_threshold=0.5,#0.7,
        )
        self.masksPatch = None
        self.bboxesPatch = None
        self.bboxesOrignal = None
        self.depths = []
        self.labels= None

        self.patchImgDepth = None

    def detectTonsil(self, image, imageDepth):
        coco_demo_tonsil=self.coco_demo_tonsil


        ###load image
        if image is None:
            image = cv2.imread('20201113_321_color44456.bmp')
            #image = cv2.imread('4_Color.png')

            #imageDepth = cv2.imread('20201113_321_depth44456.png',-1)
            #imageDepth = cv2.imread('99_Depth.png', -1)
        if imageDepth is None:
            imageDepth = cv2.imread('20201113_321_depth44456.png', -1)
            #imageDepth = cv2.imread('44_Depth.png', -1)

        #templateMatching=TemplateMatching()

        start = time.perf_counter()
        patchRect= self.mouth.getMouth(image)
        #patchRect=templateMatching.getMouthArea(image, featureMode=0, stride=1) # hog feature and downsampling using 2
        stop = time.perf_counter()
        print("嘴部区域检测时间："+str(stop - start) + "秒")

        patchImg=  image[patchRect[1]: patchRect[3], patchRect[0]: patchRect[2],:]
        patchImgDepth = imageDepth[patchRect[1]: patchRect[3], patchRect[0]: patchRect[2]]
        self.patchImgDepth = patchImgDepth
        plt.imshow(patchImg)
        plt.show()

        ###detection
        start = time.perf_counter()
        predictions,bboxesPatch,labels,masksPatch= coco_demo_tonsil.run_on_opencv_image(patchImg,detect_target='tonsil') #bbox返回坐标

        stop = time.perf_counter()

        print("扁桃体区域检测时间："+str(stop - start) + "秒")

        plt.imshow(predictions)
        plt.show()

        if bboxesPatch.device.type=="cpu":
            bboxesPatch = bboxesPatch.numpy()
            bboxesOrignal  = np.zeros(bboxesPatch.shape) ### x1,y1,x2,y2
            for i in range(bboxesPatch.shape[0]):
                bboxesOrignal[i][0] = bboxesPatch[i][0] + patchRect[0]
                bboxesOrignal[i][1] = bboxesPatch[i][1] + patchRect[1]
                bboxesOrignal[i][2] = bboxesPatch[i][2] + patchRect[0]
                bboxesOrignal[i][3] = bboxesPatch[i][3] + patchRect[1]
        if labels.device.type == "cpu":
            labels = labels.numpy()
        if masksPatch.device.type == "cpu":
            masksPatch = masksPatch.numpy()

        self.bboxesPatch = bboxesPatch
        self.masksPatch = masksPatch
        self.labels = labels
        self.bboxesOrignal = bboxesOrignal

        self._getDepth()


    def _getDepth(self):
        targetNum = self.bboxesPatch.shape[0]

        for i in range(targetNum):
            depthsList = []
            weights = []

            centerX = (self.bboxesPatch[i][0] + self.bboxesPatch[i][2]) / 2
            centerY = (self.bboxesPatch[i][1] + self.bboxesPatch[i][3]) / 2
            # if mask is true and depth >0, then added the depth value into depthList, meanwhile computing the distance between the point and the center as the weight of depth
            depth = self.patchImgDepth[int(centerY)][int(centerX)]
            if depth <= 0:
                for r in range(self.masksPatch.shape[2]):
                    for c in range(self.masksPatch.shape[3]):
                        maskTemp=self.masksPatch[i][0][r][c]
                        if maskTemp:
                            depthTemp = self.patchImgDepth[r][c]
                            if depthTemp>0:
                                depthsList.append(depthTemp)
                                distanceTemp=math.sqrt((centerX-c)**2+(centerY-r)**2)
                                weights.append(1/distanceTemp)

                #normalize weights
                weightSum=sum(weights)
                weights=np.array((weights))
                depthsList = np.array(depthsList)
                weights=weights/weightSum

                #compute the mean of depths with the weight
                depth=np.dot(depthsList,weights)
            self.depths.append(depth)
        self.depths = np.array(self.depths)



if __name__=="__main__":
    getBoxesAndDepth=GetBoxesAndDepth()
    getBoxesAndDepth.detectTonsil(None,None)
    if getBoxesAndDepth.labels.size > 0:
        print("标签:")
        print(getBoxesAndDepth.labels)
        print("坐标:")
        print(getBoxesAndDepth.bboxesOrignal)
        print("深度:")
        print(getBoxesAndDepth.depths)
    else:
        print("没有检测到目标")

