import cv2
import os
import numpy as np
from skimage import exposure
from skimage.feature import hog
from skimage import data, color, exposure

class TemplateMatching(object):
    def __init__(self):
        self.template=cv2.imread('mouthTemplate.png')

        self.templateWidth=self.template.shape[1]
        self.templateHeight=self.template.shape[0]

    def getMouthArea(self, image, featureMode=0, stride=1):
        """
        Args:
            image:
            featureMode: 0:raw pixel; 1: hog
            stride: if stride>1, there is downsampling

        Returns:
            resultRect: format with (x1,y1,x2,y2)
        """
        imageWidth=image.shape[1]
        imageHeight=image.shape[0]

        if stride >1:
            scaledWidth = int(image.shape[1]  / stride)
            scaledHeight = int(image.shape[0] / stride)
            scaledTemplateWidth =int(self.templateWidth / stride)
            scaledTemplateHeight = int(self.templateHeight / stride)

            dim = (scaledWidth, scaledHeight)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)# resize image

            dim = (scaledTemplateWidth, scaledTemplateHeight )
            self.template = cv2.resize(self.template, dim, interpolation=cv2.INTER_AREA)# resize template

        if featureMode==0:
            matchResponse=cv2.matchTemplate(image, self.template, cv2.TM_CCOEFF_NORMED)

        else:
            cellSize = 8
            orientationsNum = 16
            templateHog = hog(self.template, orientations=orientationsNum, pixels_per_cell=(cellSize, cellSize), cells_per_block=(1, 1), feature_vector=False)
            templateHog = templateHog.squeeze()
            templateHog = templateHog.astype(np.float32)

            imageHog = hog(image, orientations=orientationsNum, pixels_per_cell=(cellSize, cellSize),
                                cells_per_block=(1, 1), feature_vector=False)
            imageHog = imageHog.squeeze()
            imageHog = imageHog.astype(np.float32)

            matchResponse=[]
            matchResponseMax=[]
            for i in range(orientationsNum):
                matchResponse.append(cv2.matchTemplate(imageHog[:, :, i], templateHog[:, :, i], cv2.TM_CCOEFF_NORMED))
                matchResponseMax.append(np.max(matchResponse[i]))

            res = np.zeros(matchResponse[0].shape)

            matchResponseMaxTopNSum = 0
            topN = 4
            matchResponseTopN=[]
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

        if featureMode == 1:
            maxLoc = list(maxLoc)
            maxLoc[0] = maxLoc[0] * cellSize
            maxLoc[1] = maxLoc[1] * cellSize

        if stride > 1:
            maxLoc = list(maxLoc)
            maxLoc[0] = maxLoc[0] * stride
            maxLoc[1] = maxLoc[1] * stride

        deltaX = int(maxLoc[0]-self.templateWidth*0.25) #x1
        deltaY = int(maxLoc[1]-self.templateHeight*0.25) #y1
        resultRight = int(maxLoc[0]+self.templateWidth*1.25) #x2
        resultBottom = int(maxLoc[1]+self.templateHeight*1.25) #y2


        if resultRight > imageWidth:
            resultRight = imageWidth

        if resultBottom > imageHeight:
            resultBottom = imageHeight

        if deltaY < 0:
            deltaY = 0

        if deltaX < 0:
            deltaX = 0

        resultRect=(deltaX, deltaY, resultRight, resultBottom)#x1,y1,x2,y2
        return resultRect
