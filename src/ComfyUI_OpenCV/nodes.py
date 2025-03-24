from inspect import cleandoc
import cv2
import numpy as np
import torch


class MedianBlur:
    """
    Apply median blur filter to an image using OpenCV
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input image to apply median blur"}),
                "ksize": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 99,
                    "step": 2,  # Must be odd number
                    "display": "slider",
                    "tooltip": "Kernel size. Must be odd and greater than 1"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_median_blur"
    CATEGORY = "OpenCV/Filters"
    DESCRIPTION = cleandoc(__doc__)

    def apply_median_blur(self, image, ksize):
        # Ensure ksize is odd
        if ksize % 2 == 0:
            ksize += 1
            
        # Convert from torch tensor [B, H, W, C] to numpy array
        batch_size, height, width, channels = image.shape
        result = torch.zeros_like(image)
        
        # Process each image in the batch
        for b in range(batch_size):
            # Convert to numpy and scale to 0-255 range
            img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
            
            # Apply median blur
            blurred = cv2.medianBlur(img_np, ksize)
            
            # Convert back to torch tensor and normalize to 0-1
            result[b] = torch.from_numpy(blurred.astype(np.float32) / 255.0)
            
        return (result,)


class GaussianBlur:
    """
    Apply Gaussian blur filter to an image using OpenCV
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input image to apply Gaussian blur"}),
                "ksize_x": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 99,
                    "step": 2,  # Must be odd number
                    "display": "slider",
                    "tooltip": "Kernel width. Must be odd and greater than 1"
                }),
                "ksize_y": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 99,
                    "step": 2,  # Must be odd number
                    "display": "slider",
                    "tooltip": "Kernel height. Must be odd and greater than 1"
                }),
                "sigma_x": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Gaussian kernel standard deviation in X direction"
                }),
                "sigma_y": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Gaussian kernel standard deviation in Y direction"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_gaussian_blur"
    CATEGORY = "OpenCV/Filters"
    DESCRIPTION = cleandoc(__doc__)

    def apply_gaussian_blur(self, image, ksize_x, ksize_y, sigma_x, sigma_y):
        # Ensure ksize is odd
        if ksize_x % 2 == 0:
            ksize_x += 1
        if ksize_y % 2 == 0:
            ksize_y += 1
            
        # Convert from torch tensor [B, H, W, C] to numpy array
        batch_size, height, width, channels = image.shape
        result = torch.zeros_like(image)
        
        # Process each image in the batch
        for b in range(batch_size):
            # Convert to numpy and scale to 0-255 range
            img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img_np, (ksize_x, ksize_y), sigma_x, sigma_y)
            
            # Convert back to torch tensor and normalize to 0-1
            result[b] = torch.from_numpy(blurred.astype(np.float32) / 255.0)
            
        return (result,)


class CvtColor:
    """
    Convert image color space using OpenCV
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input image to convert color space"}),
                "code": (["BGR2GRAY", "GRAY2BGR", "BGR2RGB", "RGB2BGR", "BGR2HSV", "HSV2BGR", "BGR2Lab", "Lab2BGR"], {
                    "default": "BGR2GRAY",
                    "tooltip": "Color conversion code"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_cvt_color"
    CATEGORY = "OpenCV/Color"
    DESCRIPTION = cleandoc(__doc__)

    def apply_cvt_color(self, image, code):
        # 转换颜色空间代码映射
        code_map = {
            "BGR2GRAY": cv2.COLOR_BGR2GRAY,
            "GRAY2BGR": cv2.COLOR_GRAY2BGR,
            "BGR2RGB": cv2.COLOR_BGR2RGB,
            "RGB2BGR": cv2.COLOR_RGB2BGR,
            "BGR2HSV": cv2.COLOR_BGR2HSV,
            "HSV2BGR": cv2.COLOR_HSV2BGR,
            "BGR2Lab": cv2.COLOR_BGR2Lab,
            "Lab2BGR": cv2.COLOR_Lab2BGR
        }
        
        # 获取图像尺寸信息
        batch_size, height, width, channels = image.shape
        
        # 创建结果tensor - 移除特殊处理
        result = torch.zeros_like(image)
        
        # 处理每张图片
        for b in range(batch_size):
            # 转换为numpy数组并缩放到0-255
            img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
            
            # 应用颜色空间转换
            converted = cv2.cvtColor(img_np, code_map[code])
            
            # 如果是灰度图，扩展为3通道
            if code == "BGR2GRAY":
                converted = np.stack([converted] * 3, axis=-1)
            
            # 转回torch tensor并归一化到0-1
            result[b] = torch.from_numpy(converted.astype(np.float32) / 255.0)
            
        return (result,)



# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "OpenCV_MedianBlur": MedianBlur,
    "OpenCV_GaussianBlur": GaussianBlur,
    "OpenCV_CvtColor": CvtColor,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenCV_MedianBlur": "Median Blur",
    "OpenCV_GaussianBlur": "Gaussian Blur",
    "OpenCV_CvtColor": "Color Space Convert",
}
