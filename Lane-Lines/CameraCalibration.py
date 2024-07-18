import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class CameraCalibration():
    """
    使用棋盘格图像校准相机的类。

    属性：
        mtx (np.array): 相机矩阵
        dist (np.array): 畸变系数
    """

    def __init__(self, image_dir, nx, ny, debug=False):
        """
        初始化相机校准。

        参数：
            image_dir (str): 包含棋盘格图像的文件夹路径
            nx (int): 棋盘格的宽度（方格数）
            ny (int): 棋盘格的高度（方格数）
        """

        fnames = glob.glob("{}/*".format(image_dir))
        objpoints = []
        imgpoints = []
        
        # Coordinates of chessboard's corners in 3D
        objp = np.zeros((nx*ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        
        # Go through all chessboard images
        for f in fnames:
            img = mpimg.imread(f)

            # Convert to grayscale image
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(img, (nx, ny))
            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)

        shape = (img.shape[1], img.shape[0])
        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

        if not ret:
            raise Exception("Unable to calibrate camera")

    def undistort(self, img):
        """
        返回校正后的图像。

        参数：
            img (np.array): 输入图像

        返回：
            Image (np.array): 校正后的图像
        """

        # Convert to grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
