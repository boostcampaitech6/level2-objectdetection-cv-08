import os
from albumentations import ShiftScaleRotate, RandomBrightnessContrast, OneOf, RGBShift, HueSaturationValue, JpegCompression, ChannelShuffle, OneOf, Blur, MedianBlur
from albumentations import Resize
from shutil import copyfile
from PIL import Image
import json

# Albumentations 변환 함수 정의
def apply_albu_transform(image_path, transform, save_path):
    image = Image.open(image_path)
    augmented = transform(image=image)
    transformed_image = augmented['image']
    transformed_image.save(save_path)

# Albumentations 변환 파이프라인 정의
albu_transforms = [
    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0, rotate_limit=0, interpolation=1, p=0.5),
    RandomBrightnessContrast(brightness_limit=[0.1, 0.4], contrast_limit=[0.1, 0.2], p=0.3),
    OneOf([
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0)
    ], p=0.1),
    JpegCompression(quality_lower=85, quality_upper=95, p=0.2),
    ChannelShuffle(p=0.1),
    OneOf([
        Blur(blur_limit=3, p=1.0),
        MedianBlur(blur_limit=3, p=1.0)
    ], p=0.1)
]

# Modify dataset related settings
data_root = '../data/recycle/'
metainfo = {
    'classes': ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing',),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 230), (106, 0, 228), (60, 20, 220),
        (0, 80, 100), (0, 0, 70), (50, 0, 192), (250, 170, 30), (255, 0, 0)
    ]
}

# 훈련 데이터 경로 및 파일 목록 가져오기
train_data_path = os.path.join(data_root,  'train')
train_json_path = os.path.join(data_root, 'train.json')

with open(train_json_path, 'r') as f:
    train_data = json.load(f)

transform_folder = os.path.join(data_root,'transform')
os.makedirs(transform_folder, exist_ok=True)

# 'transform' 폴더에 Albumentations을 적용하여 이미지 변환 및 저장
transformed_annotations = {'images': [], 'annotations': []}

for img_info in train_data['images']:
    img_filename = img_info['file_name']
    img_path = os.path.join(train_data_path, img_filename)
    save_path = os.path.join(transform_folder, img_filename)

    # Albumentations 변환 적용
    for albu_transform in albu_transforms:
        transformed_save_path = save_path.replace('.jpg', f'_transformed_{albu_transform.__class__.__name__}.jpg')
        apply_albu_transform(img_path, albu_transform, transformed_save_path)

        # 새로운 이미지 정보 생성 및 저장된 이미지의 annotations 생성
        transformed_annotations['images'].append({
            'file_name': os.path.basename(transformed_save_path),
            'id': len(transformed_annotations['images']) + 1
        })

        transformed_annotations['annotations'].append({
            'image_id': len(transformed_annotations['images']),
            'bbox': [0, 0, 100, 100],  # 적절한 bounding box 정보로 대체해야 함
            'category_id': 1,  # 적절한 category_id로 대체해야 함
            'id': len(transformed_annotations['annotations']) + 1
        })

# 'transform' 폴더에 저장된 annotations를 transformed_annotations 파일로 저장
transformed_json_path = os.path.join(data_root, 'transform.json')
with open(transformed_json_path, 'w') as f:
    json.dump(transformed_annotations, f)

