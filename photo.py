import pyrealsense2 as rs
import numpy as np
import cv2
# import scipy.misc
import time


def get_image(obj_num):

    # 初始化相机对象
    config = rs.config()
    # 深度初始化
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    # 颜色初始化
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # 红外初始化（左右）
    config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)

    # 创建数据流
    pipeline = rs.pipeline()
    pipeline.start(config)

    align = rs.align(rs.stream.color)
    # 获取图像, realsense刚启动的时候图像会有一些失真，我们保存第100帧图片。
    for i in range(200):
        frames = pipeline.wait_for_frames()
        data = align.process(frames)
        depth = data.get_depth_frame()
        color = data.get_color_frame()
        ir_left = data.get_infrared_frame(1)
        ir_right = data.get_infrared_frame(2)
        if not depth or not color:
            continue

    ######################## 获取内参 get_profile ##################

    # 颜色摄像头内参
    cprofile = color.get_profile()
    cvsprofile = rs.video_stream_profile(cprofile)
    color_intrin = cvsprofile.get_intrinsics()
    print(color_intrin)

    # 深度摄像头内参
    dprofile = depth.get_profile()
    dvsprofile = rs.video_stream_profile(dprofile)
    depth_intrin = dvsprofile.get_intrinsics()
    print(depth_intrin)

    # 红外摄像头内参
    lprofile = ir_left.get_profile()
    rprofile = ir_right.get_profile()
    lvsprofile = rs.video_stream_profile(lprofile)
    rvsprofile = rs.video_stream_profile(rprofile)
    ir_left_intrin = lvsprofile.get_intrinsics()
    print(ir_left_intrin)
    ir_right_intrin = rvsprofile.get_intrinsics()
    print(ir_right_intrin)
    #############################################################

    #外参
    # extrin = dprofile.get_extrinsics_to(cprofile)
    # print(extrin)
    extrin = lprofile.get_extrinsics_to(rprofile)
    print(extrin)

    depth_image = np.asanyarray(depth.get_data())
    color_image = np.asanyarray(color.get_data())
    ir_left_image = np.asanyarray(ir_left.get_data())
    ir_right_image = np.asanyarray(ir_right.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.4), cv2.COLORMAP_JET)

    # Show images
    images0 = np.hstack((ir_left_image, ir_right_image))
    # cv2.imshow('images0', images0)
    # cv2.imshow('ir_left_image', ir_left_image)
    # cv2.imshow('ir_right_image', ir_right_image)
    cv2.waitKey(100)

    t = time.time()
    tname = str(t)[5:10]

    depth_file = '%ddepth.png' % obj_num
    rgb_file = '%drgb.bmp' % obj_num
    cv2.imwrite(depth_file, depth_image)
    cv2.imwrite(rgb_file, color_image)

    return depth_file, rgb_file


    #cv2.imwrite('ir_left_image' + str(tname) + '.png', ir_left_image)
    #cv2.imwrite('ir_right_image' + str(tname) + '.png', ir_right_image)

    #obj_num = '_box2_'
    #cv2.imwrite('./depth_process/' + 'pcd%sr.png'%(obj_num), color_image)
    #cv2.imwrite('./depth_process/' + 'pcd%sd.png'%(obj_num), depth_image)

    #cv2.imwrite('depth_colorMAP' + str(tname) + '.png', depth_colormap)
    #scipy.misc.imsave('outfile1.png', depth_image)
    #scipy.misc.imsave('outfile2.png', color_image)

