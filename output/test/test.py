import tensorflow as tf
import tensorflow.keras as keras
import sys
sys.modules['keras'] = keras

import os
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.config import Config

# 设置根目录
ROOT_DIR = os.path.abspath("")

# 定义配置
class LeavesConfig(Config):
    NAME = "leaves"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # 背景 + 叶片
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

# 实例化配置
config = LeavesConfig()

# 创建 Mask R-CNN 模型
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=ROOT_DIR)

# 加载预训练的权重
model_path = "mask_rcnn_coco.h5"
model.load_weights(model_path, by_name=True)

# 加载白菜图像
image = cv2.imread("/mnt/data/centered_22A-T-36-10球形侧视图.JPG")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 执行检测
results = model.detect([image], verbose=1)

# 获得结果
r = results[0]

# 可视化检测到的叶片
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            ['BG', 'leaves'], r['scores'])

# 分割每片叶片
masks = r['masks']
num_leaves = masks.shape[-1]
for i in range(num_leaves):
    mask = masks[:, :, i]
    # 生成每片叶片的掩码图像
    leaf_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8)*255)
    
    # 保存每个叶片的图像
    output_path = f"leaf_{i}.png"
    cv2.imwrite(output_path, cv2.cvtColor(leaf_image, cv2.COLOR_RGB2BGR))
    print(f"Leaf {i} saved to {output_path}")