import argparse

import cv2
import os
import numpy as np

from skimage import exposure
from skimage.feature import hog
from skimage import data, color, exposure
from predictor import COCODemo
import torch
from maskrcnn_benchmark.config import cfg



class Mouth(object):
    def __init__(self, usingNetwork=True):
        self.usingNetwork=usingNetwork
        self.template = cv2.imread('mouthTemplate.png')

        self.templateWidth = self.template.shape[1]
        self.templateHeight = self.template.shape[0]

        #re-write
        if usingNetwork:
            #initialize model
            self.cfg=cfg.clone()
            config_file="../configs/my_mouth_test_e2e_mask_rcnn_R_50_FPN_1x.yaml"
            # config_file="../configs/my_mouth_test_e2e_mask_rcnn_R_50_C4_1x.yaml"

            # update the config options with the config file
            self.cfg.merge_from_file(config_file)

            # manual override some options
            self.cfg.merge_from_list(["MODEL.DEVICE", "cuda"])  ###cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

            self.coco_demo = COCODemo(
                self.cfg,
                min_image_size=800,  # 400,#800,
                confidence_threshold=0.7,  # 0.7,
            )


    def getMouth(self, image, featureMode=0):
        if not self.usingNetwork  :
            return self._getMouthUsingTemplate(self,image)
        else:
            return self._getMouthUsingMaskRcnn(image)


    def _getMouthUsingTemplate(self,image,model,featureMode=0):
        if featureMode==0:
            cellSize = 8
            orientationsNum=16
            templateHog = hog(self.template, orientations=orientationsNum, pixels_per_cell=(cellSize, cellSize),
                              cells_per_block=(1, 1), feature_vector=False)
            templateHog = templateHog.squeeze()
            templateHog = templateHog.astype(np.float32)

            imageHog = hog(image, orientations=orientationsNum, pixels_per_cell=(cellSize, cellSize),
                           cells_per_block=(1, 1), feature_vector=False)
            imageHog = imageHog.squeeze()
            imageHog = imageHog.astype(np.float32)

            matchResponse = []
            matchResponseMax = []
            for i in range(orientationsNum):
                matchResponse.append(cv2.matchTemplate(imageHog[:, :, i], templateHog[:, :, i], cv2.TM_CCOEFF_NORMED))
                matchResponseMax.append(np.max(matchResponse[i]))

            res = np.zeros(matchResponse[0].shape)

            matchResponseMaxTopNSum = 0
            topN = 4
            matchResponseTopN = []
            for i in range(topN):
                resIndex = matchResponseMax.index(max(matchResponseMax))

                matchResponseTopN.append(matchResponse[resIndex])
                del matchResponse[resIndex]
                matchResponseMaxTopNSum += matchResponseMax[resIndex]
                del matchResponseMax[resIndex]
                res = res + matchResponseTopN[i]

            matchResponse = res / topN

        cv2.normalize(matchResponse, matchResponse, 1, 0, cv2.NORM_MINMAX)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(matchResponse)

        if featureMode==1:
            maxLoc=list(maxLoc)
            maxLoc[0] = maxLoc[0] * cellSize
            maxLoc[1] = maxLoc[1] * cellSize
        deltaX = int(maxLoc[0]-self.templateWidth*0.25)
        deltaY = int(maxLoc[1]-self.templateHeight*0.25)
        resultRight = int(maxLoc[0]+self.templateWidth*1.25)
        resultBottom = int(maxLoc[1]+self.templateHeight*1.25)

        imageWidth=image.shape[1]
        imageHeight=image.shape[0]
        if resultRight > imageWidth:
            resultRight = imageWidth

        if resultBottom > imageHeight:
            resultBottom = imageHeight

        if deltaY < 0:
            deltaY = 0

        if deltaX < 0:
            deltaX = 0


        resultRect=(deltaX, deltaY, resultRight, resultBottom)#x1,y1,x2,y2
        #resultPatch = image[deltaY: resultBottom, deltaX: resultRight]
        # cv2.namedWindow("resultPatch",cv2.WINDOW_NORMAL)
        # cv2.imshow("resultPatch",resultPatch)
        # cv2.waitKey(0)

        return resultRect

    def _getMouthUsingMaskRcnn(self,image):
        coco_demo=self.coco_demo

        _,bboxesPatch,_,_= coco_demo.run_on_opencv_image(image)
        imageWidth=image.shape[1]
        imageHeight=image.shape[0]
        x1=0
        x2=0
        y1=0
        y2=0
        if bboxesPatch.shape[0]==1:
            x1,y1,x2,y2=bboxesPatch[0]
            x1 = x1.item()
            y1 = y1.item()
            x2 = x2.item()
            y2 = y2.item()
            width=x2-x1
            height=y2-y1
            padding_ratio=0.1
            x1 = int(x1 - width * padding_ratio)  # x1
            y1 = int(y1 - height * padding_ratio)  # y1
            x2 = int(x2 + width * padding_ratio)  # x2
            y2 = int(y2 + height * padding_ratio)  # y2

        if x2 > imageWidth:
            x2 = imageWidth

        if y2 > imageHeight:
            y2 = imageHeight

        if x1 < 0:
            x1 = 0

        if y1 < 0:
            y1 = 0

        resultRect = (x1, y1, x2, y2)  # x1,y1,x2,y2
        return resultRect


    def processImages(self,path,model):
        #path = ('D:/XM/datasets/Image20201113/Oringinal/')
        path_list = os.listdir(path)
        for image in path_list:
            # index+=1
            image = path + image
            pt1,pt2 =self.getMouth(image,2)
            img_cpy = cv2.imread(image, 1)
            top_edge = pt1[1]#左上点的纵坐标，就是截图的上边
            bottom_edge = pt2[1]#右下点的纵坐标
            left_edge=pt1[0]#左上点的横坐标
            right_edge=pt2[0]#右下点的横坐标
            width = right_edge-left_edge
            height=top_edge-bottom_edge
            roi = img_cpy[top_edge-0.25*height : bottom_edge+0.25*height, left_edge-0.25*width: right_edge+0.25*width]
            cv2.imshow("image_roi", roi)
            roi_path = 'D:/XM/datasets/Image20201113/Crop/crop_hog_match1/' + '202011' + format(str(cnt),
                                                                                                '0>6s') + '.png'
            cnt = cnt + 1
            cv2.imwrite(roi_path, roi)
            cv2.waitKey(1)
