from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2
import os
import time
import numpy as np

class Detect4CCode(object):

    def __init__(self):
        print("Begining to initialize Detect4CCode instance...")
        config_file = "C:/codes/maskrcnn_benchmark/maskrcnn-benchmark/configs/my_test_e2e_mask_rcnn_R_50_FPN_1x.yaml"

        # update the config options with the config file
        cfg.merge_from_file(config_file)

        # manual override some options
        cfg.merge_from_list(["MODEL.DEVICE", "cuda"])###cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

        self.coco_demo = COCODemo(
            cfg,
            min_image_size=800,#400,#800,
            confidence_threshold=0.5,#0.7,
        )
        print("读取MaskRcnn模型成功！")

    def DetectTonsil(self, image):
        # print(imageDepth)
        # print(image)
        print("begin to read coco_demo...")
        coco_demo=self.coco_demo

        #load image and then run prediction

        print("begin to detect...")
        if image is None:
            print("image is none, now loading it...")
            image =cv2.imread('202011000000.png')
            # imageDepth = cv2.imread('D202011000000.png',-1)
            #
            # print(imageDepth)
            # print(image)
        # if imageDepth is None:
        #     print("imageDepth is none!!")
        start = time.clock()
        predictions,bbox,contours,labels= coco_demo.run_on_opencv_image(image) #bbox返回坐标
        stop = time.clock()

        print(str(stop - start) + "秒")

        cv2.namedWindow("predections",cv2.WINDOW_NORMAL)
        cv2.imshow("predections",predictions)
        cv2.waitKey(0)
        # cv2.imwrite("x1xxx.jpg",predictions)
        if bbox.device.type=="cpu":
            bbox=bbox.numpy()
        if labels.device.type == "cpu":
                labels = labels.numpy()


        detectNum=labels.size
        detectRes=np.zeros((detectNum,6),dtype=float) # 每一行的格式是： 检测到的物体数量，label,x1,y1,x2,y2,xCenter,yCenter,depthCenter
        detectRes[:,0]=detectNum
        detectRes[:,1]=labels
        detectRes[:,2:6]=bbox
        # depths=[]
        # for i in range(detectNum):
        #     cols=int(bbox[i][2])-int(bbox[i][0])
        #     rows = int(bbox[i][3]) - int(bbox[i][1])
        #     r=int(bbox[i][1])
        #     c=int(bbox[i][0])
        #     count=0
        #     depthSum=0
        #     for j in range(cols):
        #         for k in range(rows):
        #             if imageDepth[j][k]>0:
        #                 count+=1
        #                 depthSum=depthSum+imageDepth[j][k]
        #     if count>0:
        #         depthSum=depthSum/count
        #     depths.append(depthSum)
        # detectRes[:, 6] = depths
        #print(detectRes)
        return detectRes

if __name__=="__main__":
    detect4CCode=Detect4CCode()
    detect4CCode.DetectTonsil(None,None)