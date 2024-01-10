import json
import mmcv
import matplotlib.pyplot as plt
from mmdet.structures.bbox import __init__
from mmdet.datasets.transforms import transforms
from mmdet.structures.bbox import base_boxes
from PIL import Image
import random
import matplotlib.patches as patches

def visualize_transformations(json_path, img_root, transform_classes, num_rows=5, num_cols=4):
    # Load the JSON file containing image and annotation information
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # Load an example image using the first image file_name in the JSON data
    img_info = random.choice(json_data['images'])
    img_path = img_root + img_info['file_name']
    img = mmcv.imread(img_path)

    # Get ground truth bounding boxes and labels for the image
    image_id = img_info['id']
    gt_bboxes = []
    gt_labels = []

    for annotation in json_data['annotations']:
        if annotation['image_id'] == image_id:
            gt_bboxes.append(annotation['bbox'])
            gt_labels.append(annotation['category_id'])

    # Convert gt_bboxes to Boxes format
    gt_bboxes = Boxes(gt_bboxes)

    # Add ground truth information to results with Boxes
    dummy_results = {'img': img, 'gt_bboxes': gt_bboxes, 'gt_labels': gt_labels}

    # Remove incompatible transform classes
    transform_classes = [transform_class for transform_class in transform_classes if
                         'SegRescale' not in transform_class.__class__.__name__]

    # Configure the plot layout
    plt.figure(figsize=(num_cols * 5, num_rows * 5))

    # Plot the original image
    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.title('Original Image')
    plt.axis('off')

    # Plot each transformed image
    for i, transform_class in enumerate(transform_classes, start=2):
        try:
            # Pass the dummy_results instead of img directly
            transformed_results = transform_class(dummy_results)
            transformed_img = transformed_results['img']
            transformed_gt_bboxes = transformed_results['gt_bboxes'].tensor.numpy()

            plt.subplot(num_rows, num_cols, i)
            plt.imshow(mmcv.bgr2rgb(transformed_img))
            plt.title(transform_class.__class__.__name__)
            plt.axis('off')

            # Visualize transformed bounding boxes
            for bbox in transformed_gt_bboxes:
                plt.gca().add_patch(patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                    fill=False, edgecolor='red', linewidth=2
                ))
        except Exception as e:
            print(f"Error in {transform_class.__class__.__name__}: {e}")

    plt.tight_layout()
    plt.show()

# Example Usage
json_path = '../data/recycle/train.json'
img_root = '../data/recycle/'

# Define the Albumentations transforms
albu_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

# Define the mmdet transforms (without RandomCenterCropPad)
transform_classes = [
    transforms.RandomFlip(prob=1.0),
    transforms.RandomShift(),
    transforms.Pad(size=(512, 512)),
    transforms.PhotoMetricDistortion(),
    transforms.Expand(),
    transforms.Corrupt(corruption='gaussian_noise'),
    transforms.Albu(transforms=albu_transforms),
    transforms.CutOut(n_holes=8, cutout_shape=(50, 50)),
    transforms.Mosaic(),
    transforms.MixUp(),
    transforms.RandomAffine(),
    transforms.YOLOXHSVRandomAug(),
    transforms.CopyPaste()
]

# Visualize the transformations
visualize_transformations(json_path, img_root, transform_classes)
