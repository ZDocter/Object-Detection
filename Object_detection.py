"""
======================XOC数据集下载=======================
2012下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
2007下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
"""

'''======================看数据集标注=======================查'''
# SegmentationClass、SegmentationObject 用于语义分割
# ImageSets中的Main中文件为样本标注

"""
======================COCO数据集=======================
"""

"""
======================标注自己的数据集=======================
在线标注网站：https://www.makesense.ai/
https://www.cvat.org/
本地标记软件：精灵标注助手
"""

"""
======================使用代码加载数据集=======================
                    pytorch读取COCO数据集
"""
import torch
import numpy as np
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from PIL import ImageDraw


# Load dataset
file_root = r'D:\PyTorch_Python\Object_detection_learning\dataset\val2017'
ann_root = r'D:\PyTorch_Python\Object_detection_learning\dataset\annotations\instances_val2017.json'
coco_dataset = CocoDetection(root=file_root, annFile=ann_root)
image, info = coco_dataset[0]
image_handler = ImageDraw.ImageDraw(image)

# Get bbox
for annotation in info:
    x_min, y_min, width, height = annotation['bbox']
    # Draw a rectangle
    image_handler.rectangle(((x_min, y_min), (x_min+width, y_min+height)))
image.show()