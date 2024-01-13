import os
import random
import numpy as np
import cv2
import torch
import json
from PIL import Image, ImageDraw
import matplotlib.patches as patches


def xywh2xyxy(x, w=1024, h=1024, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywh(x, w=1024, h=1024):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def noPad(label, xc, yc, xmin, ymin, s):
    label4_ = label.copy()
    if (xc > s) & (yc < s):
        label4_[:, 1] = label4_[:, 1] - xmin
        label4_[:, 3] = label4_[:, 3] - xmin
    elif (xc < s) & (yc > s):
        label4_[:, 2] = label4_[:, 2] - ymin
        label4_[:, 4] = label4_[:, 4] - ymin
    elif (xc > s) & (yc > s):
        label4_[:, 1] = label4_[:, 1] - xmin
        label4_[:, 3] = label4_[:, 3] - xmin
        label4_[:, 2] = label4_[:, 2] - ymin
        label4_[:, 4] = label4_[:, 4] - ymin
    return label4_


def myFig(img, bbox, drawB=True):
    image = Image.fromarray(img, "RGB")
    draw = ImageDraw.Draw(image)

    if drawB:
        for i in range(len(bbox)):
            xmin = bbox[i][1]
            ymin = bbox[i][2]
            xmax = bbox[i][3]
            ymax = bbox[i][4]

            draw.rectangle(
                (xmin, ymin, xmax, ymax), outline=(255, 0, 0), width=1
            )  # bounding box
    #     display(image)
    return image


def Mosaic(img_folder, indices, annotations):
    labels4 = []
    img_size = 1024
    s = img_size
    mosaic_border = [-img_size // 2, -img_size // 2]

    # center point
    yc, xc = (
        int(random.uniform(-x, 2 * s + x)) for x in mosaic_border
    )  # mosaic center x, y # 81 172

    for i, index in enumerate(indices):
        # Load image
        img = cv2.imread(os.path.join(img_folder, index))
        h, w = 1024, 1024

        # place img in img4
        if i == 0:  # top left
            img4 = np.full(
                (s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8
            )  # base image with 4 tiles
            x1a, y1a, x2a, y2a = (
                max(xc - w, 0),
                max(yc - h, 0),
                xc,
                yc,
            )  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = (
                w - (x2a - x1a),
                h - (y2a - y1a),
                w,
                h,
            )  # xmin, ymin, xmax, ymax (small image)

            xmin, ymin = x1a, y1a

        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            xmax, ymax = x2a, y2a

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels = []
        for anno in annotations[i]:
            labels.append([anno["category_id"], *anno["bbox"]])  # xywh
        labels = np.array(labels)

        if labels.size:
            labels[:, 1:] = xywh2xyxy(
                labels[:, 1:], w, h, padw, padh
            )  # normalized xywh to pixel xyxy format
            labels[labels < 0] = 0
            labels[labels > s * 2] = s * 2
        labels4.append(labels)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)

    # no padding
    img4 = img4[ymin:ymax, xmin:xmax, :]
    labels4 = noPad(labels4, xc, yc, xmin, ymin, s)

    # resize
    img4 = cv2.resize(img4, (w, h))
    labels4[:, 1:] = xyxy2xywh(labels4[:, 1:], w=xmax - xmin, h=ymax - ymin)
    labels4 = labels4[labels4[:, 3] > 0]
    labels4 = labels4[labels4[:, 4] > 0]

    return img4, labels4


if __name__ == "__main__":
    # image files and labels
    # train set & not Scratch
    data_path = "/data/ephemeral/home/level2-objectdetection-cv-08/data/recycle"
    img_folder = os.path.join(data_path, "train")
    im_files = [i for i in os.listdir(img_folder) if i.endswith("jpg")]

    # random 4 images
    indices = random.choices(
        im_files, k=4
    )  # 3 additional image indices # [0, 3292, 20762, 18713]
    random.shuffle(indices)  # [18713, 0, 20762, 3292]

    # train.json 파일 경로
    json_path = os.path.join(data_path, "train.json")

    # JSON 파일에서 이미지 및 바운딩 박스 정보 읽기
    with open(json_path, "r") as json_file:
        data = json.load(json_file)

    annotations = []
    for index in indices:
        annotations.append(
            [anno for anno in data["annotations"] if anno["image_id"] == int(index[:4])]
        )

    img4, labels4 = Mosaic(img_folder, indices, annotations)

    img = Image.fromarray(img4)
    img.save(
        "/data/ephemeral/home/level2-objectdetection-cv-08/Upsampling/mosaic_sample.jpg",
        "JPEG",
    )

    img_bbox = myFig(img4, labels4)
    img_bbox.save(
        "/data/ephemeral/home/level2-objectdetection-cv-08/Upsampling/mosaic_sample_bbox.jpg",
        "JPEG",
    )
