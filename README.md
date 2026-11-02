无人机垃圾检测数据集![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/be74a4d09cbb4317872eae40936794cf.png)


772张无人机航拍图像，3718个标注框。近地环境下拍摄，所包含垃圾的图像中，目标框都较小。图像为3840×2160，总共标记为1类，即 rubbish，环境垃圾![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f69e6c51c2ea489a8f0be64bd2a6c21c.png)


coco格式，可通过pycocotools中的coco工具进行操作，

# 使用YOLOv8训练无人机垃圾检测数据集

## 一、环境搭建

### 1. 安装CUDA驱动
```bash
# 检查当前CUDA版本
nvidia-smi

# 如果需要安装特定版本（以Ubuntu为例）
sudo apt-get install cuda-12-1
```

### 2. 安装Anaconda
```bash
# 下载Anaconda安装包
curl -O https://repo.anaconda.com/archive/Anaconda3-2023.07-Linux-x86_64.sh

# 安装Anaconda
bash Anaconda3-2023.07-Linux-x86_64.sh

# 初始化conda环境
source ~/.bashrc
```

### 3. 创建Python虚拟环境
```bash
# 创建新环境
conda create -n uavwaste python=3.9
conda activate uavwaste

# 安装PyTorch和YOLOv8依赖
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python pycocotools matplotlib numpy pandas openpyxl
```

## 二、数据准备

### UAVVaste数据集结构:
```
UAVVaste/
├── images/
│   ├── train/
│   └── val/
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

### COCO格式转换为YOLOv8格式的脚本:
```python
import json
import os
from pathlib import Path

def coco_to_yolo(coco_json_path, images_dir, output_dir):
    # 加载COCO标注文件
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取类别信息（UAVVaste只有一个类别'rubbish'）
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f"发现类别: {categories}")
    
    # 创建图像ID到文件名的映射
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # 处理每个图像的标注
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        file_name = image_id_to_filename[image_id]
        
        # 获取图像尺寸
        image_path = os.path.join(images_dir, file_name)
        if not os.path.exists(image_path):
            print(f"警告: 图像文件不存在 - {image_path}")
            continue
            
        with open(image_path, 'rb') as img_file:
            from PIL import Image
            img = Image.open(img_file)
            img_w, img_h = img.size
        
        # 转换COCO格式的bbox [x, y, w, h] 到YOLOv5格式的归一化坐标
        x, y, w, h = ann['bbox']
        xc = (x + w/2) / img_w
        yc = (y + h/2) / img_h
        wn = w / img_w
        hn = h / img_h
        
        # 创建标签文件
        label_file = output_dir / Path(file_name).with_suffix('.txt')
        with open(label_file, 'a') as f:
            # 对于UAVWaste数据集，所有目标都是类别0
            f.write(f"0 {xc} {yc} {wn} {hn}\n")
    
    print("转换完成!")

# 转换训练集
coco_to_yolo(
    coco_json_path='UAVVaste/annotations/instances_train.json',
    images_dir='UAVVaste/images/train',
    output_dir='UAVVaste/labels/train'
)

# 转换验证集
coco_to_yolo(
    coco_json_path='UAVVaste/annotations/instances_val.json',
    images_dir='UAVVaste/images/val',
    output_dir='UAVVaste/labels/val'
)
```

## 三、创建data.yaml配置文件

```yaml
train: ./UAVVaste/images/train
val: ./UAVVaste/images/val

nc: 1
names: ['rubbish']
```

## 四、模型训练

### 下载YOLOv8预训练权重：
```bash
# 下载YOLOv8n模型（轻量级）
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### 训练脚本：
```python
from ultralytics import YOLO
import datetime

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 获取当前时间作为训练标识
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 开始训练
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name=f'uavwaste_{current_time}',
    project='runs/uavwaste',
    pretrained=True,
    optimizer='AdamW',  # 使用AdamW优化器更适合小目标检测
    lr0=0.01,          # 初始学习率
    lrf=0.01,          # 最终学习率
    momentum=0.937,    # 动量参数
    weight_decay=0.0005,  # 权重衰减
    warmup_epochs=3,    # 预热期
    box=0.05,           # box损失权重
    cls=0.5,            # 分类损失权重
    dfl=1.5,            # DFL损失权重
    hsv_h=0.015,        # 色彩增强参数
    hsv_s=0.7, 
    hsv_v=0.4,
    degrees=0,          # 不使用旋转增强
    translate=0.1,      # 平移增强
    scale=0.5,          # 缩放增强
    shear=0,            # 剪切增强
    perspective=0.0001, # 透视变换
    flipud=0.5,         # 上下翻转
    fliplr=0.5,         # 左右翻转
    mosaic=1.0,         # 马赛克增强
    mixup=0.2,          # MixUp增强
    copy_paste=0.3,     # Copy-Paste增强
    close_mosaic=10,    # 关闭马赛克增强的epoch数
    verbose=True
)
```

## 五、模型验证

```python
# 加载最佳模型进行验证
best_model = YOLO('runs/uavwaste/uavwaste_20230815_143045/best.pt')

# 在验证集上评估模型性能
metrics = best_model.val()

# 打印主要指标
print(f"mAP@0.5-0.95: {metrics.box.map:.4f}")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"Recall@max1: {metrics.box.recall[0]:.4f}")
print(f"Precision@max1: {metrics.box.precision[0]:.4f}")

# 可视化一些验证结果
results = best_model('UAVVaste/images/val/', stream=True)
for i, r in enumerate(results):
    if i < 5:  # 显示前5个验证结果
        r.plot()
        plt.show()
```

## 六、模型推理与部署

### 单张图片推理：
```python
# 对单张图片进行推理
results = best_model('UAVVaste/images/val/example.jpg')

# 显示结果
results[0].plot()
plt.show()
```

### 视频流实时检测：
```python
import cv2

# 打开视频文件或摄像头
cap = cv2.VideoCapture('video.mp4')  # 或使用0表示默认摄像头

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 对当前帧进行推理
    results = best_model(frame)
    
    # 绘制检测结果
    annotated_frame = results[0].plot()
    
    # 显示结果
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    
    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 七、针对小目标检测的优化建议

1. **多尺度预测**：YOLOv8默认使用多尺度预测，但可以调整PANet中的特征融合方式来更好地处理小目标。

2. **改进特征提取网络**：
   ```python
   # 在训练时使用更强大的骨干网络
   model = YOLO('yolov8m.pt')  # 使用中等大小的模型代替小型模型
   ```

3. **增加特征金字塔的层级**：
   ```python
   # 修改模型配置文件(yolov8.yaml)，添加更多特征金字塔层级
   # 这需要自定义模型配置，适用于高级用户
   ```

4. **优化数据增强策略**：
   ```python
   # 在训练参数中增加对小目标友好的增强方法
   results = model.train(
       ...
       scale=0.8,         # 更大的缩放范围
       crop_fraction=0.3, # 添加随机裁剪
       erasing=0.4,       # 添加随机擦除
       ...
   )
   ```

5. **调整锚框尺寸**：
   ```python
   # 自动计算适合UAVWaste数据集的锚框
   from ultralytics.utils.autoanchor import check_anchors
   check_anchors(dataset, thr=hyp['anchor_t'], imgsz=imgsz)
   ```

6. **使用高分辨率输入**：
   ```python
   # 将输入图像大小从默认的640提高到1280
   results = model.train(
       ...
       imgsz=1280,
       ...
   )
   ```

7. **采用注意力机制**：
   ```python
   # 使用集成CBAM注意力模块的YOLO变体
   # 这需要修改模型配置文件，适用于高级用户
   ```


