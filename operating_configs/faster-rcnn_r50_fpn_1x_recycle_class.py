# The new config inherits a base config to highlight the necessary modification
_base_ = "../mmdetection/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco.py"

# We also need to change the num_classes in head to match the dataset's annotation
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=1), mask_head=dict(num_classes=1)))

custom_hooks = [
    dict(type="SubmissionHook", test_out_dir="submit"),
    dict(type="MetricHook"),
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=10,
        )
    )
)


# Modify dataset related settings
data_root = "/home/hojun/Documents/code/boostcamp/project2/version1/dataset/"
metainfo = {
    "classes": (
        "General trash",
        "Paper",
        "Paper pack",
        "Metal",
        "Glass",
        "Plastic",
        "Styrofoam",
        "Plastic bag",
        "Battery",
        "Clothing",
    ),
    "palette": [
        (220, 20, 60),
        (119, 11, 32),
        (0, 0, 230),
        (106, 0, 228),
        (60, 20, 220),
        (0, 80, 100),
        (0, 0, 70),
        (50, 0, 192),
        (250, 170, 30),
        (255, 0, 0),
    ],
}


train_dataloader = dict(
    batch_size=16,
    num_workers=12,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="train_eye_eda.json",
        data_prefix=dict(img=""),
    ),
)
val_dataloader = dict(
    num_workers=12,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="val_eye_eda.json",
        data_prefix=dict(img=""),
    ),
)
test_dataloader = dict(
    num_workers=12,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="test.json",
        data_prefix=dict(img=""),
    ),
)


# Modify metric related settings
val_evaluator = dict(ann_file=data_root + "val_eye_eda.json", classwise=True)
test_evaluator = dict(ann_file=data_root + "test.json")

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"