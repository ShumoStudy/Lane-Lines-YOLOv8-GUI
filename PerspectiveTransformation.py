import cv2
import numpy as np

class PerspectiveTransformation:
    """
    这是一个用于图像在正视图和俯视图之间转换的类。

    属性：
        src (np.array): 4个源点的坐标
        dst (np.array): 4个目标点的坐标
        M (np.array): 从正视图到俯视图转换图像的矩阵
        M_inv (np.array): 从俯视图到正视图转换图像的矩阵
    """

    def __init__(self):
        """初始化透视变换"""
        self.src = np.float32([(550, 460),     # top-left
                               (150, 720),     # bottom-left
                               (1200, 720),    # bottom-right
                               (770, 460)])    # top-right
        self.dst = np.float32([(100, 0),
                               (100, 720),
                               (1100, 720),
                               (1100, 0)])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def forward(self, img, img_size=(1280, 720), flags=cv2.INTER_LINEAR):
        """
        接收一个正视图图像并将其转换为俯视图。

        参数：
            img (np.array): 一个正视图图像
            img_size (tuple): 图像的尺寸（宽度，高度）
            flags: 在cv2.warpPerspective()中使用的标志

        返回：
            Image (np.array): 俯视图图像
        """

        return cv2.warpPerspective(img, self.M, img_size, flags=flags)

    def backward(self, img, img_size=(1280, 720), flags=cv2.INTER_LINEAR):
        """
        接收一个俯视图图像并将其转换为正视图。

        参数：
            img (np.array): 一个俯视图图像
            img_size (tuple): 图像的尺寸（宽度，高度）
            flags (int): 在cv2.warpPerspective()中使用的标志

        返回：
            Image (np.array): 正视图图像
        """

        return cv2.warpPerspective(img, self.M_inv, img_size, flags=flags)
