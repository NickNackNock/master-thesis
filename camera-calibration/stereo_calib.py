"""
Copyright: most of the code here is from the following Guide:
Stereo Camera Calibration and Triangulation with OpenCV and Python (https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html)
By: Temuge Batpurev Feb 2, 2021
"""
import cv2
import glob
import numpy as np
def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
    #read the synched frames
    images_names = glob.glob(frames_folder)
    images_names = sorted(images_names)

    c1_images_names = sorted(glob.glob(frames_folder.replace('*', '*_first_camera*')))
    c2_images_names = sorted(glob.glob(frames_folder.replace('*', '*_second_camera*')))
 
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv2.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv2.imread(im2, 1)
        c2_images.append(_im)
 
    #change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
    rows = 5 #number of checkerboard rows.
    columns = 8 #number of checkerboard columns.
    world_scaling = 1. #change this to the real world square size. Or not.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv2.findChessboardCorners(gray1, (5, 8), None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, (5, 8), None)
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
            cv2.drawChessboardCorners(frame1, (5,8), corners1, c_ret1)
            cv2.imshow('img', frame1)
 
            cv2.drawChessboardCorners(frame2, (5,8), corners2, c_ret2)
            cv2.imshow('img2', frame2)
            k = cv2.waitKey(500)
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
 
    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
    
    print("Optimal values of 'ret' should be below 1")
    print("The MSE on reconstruction is: ",ret)
    return R, T

def load_camera_params(folder):
    mtx  = np.loadtxt(f"./{folder}/camera_matrix.txt")
    dist = np.loadtxt(f"./{folder}/distortion_coefficients.txt")
    return mtx, dist


mtx1, dist1 = load_camera_params("output_first_camera")
mtx2, dist2 = load_camera_params("output_second_camera")
R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, './synced/*')
