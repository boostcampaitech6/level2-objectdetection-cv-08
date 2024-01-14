# The new config inherits a base config to highlight the necessary modification
_base_ = ""

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    data_preprocessor=dict(
        mean=[123.65, 117.40, 110.07],
        std=[54.01, 53.36, 54.77],
    ),
)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
default_hooks = dict(
    visualization=dict(type="DetVisualizationHook", draw=True, interval=20)
)

custom_hooks = [
    dict(type="SubmissionHook"),
    dict(type="MetricHook"),
]

# Modify dataset related settings
data_root = "data/recycle/"
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

# Modify data loaders and evaluators
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="train_eye_eda.json",
        data_prefix=dict(img=""),
    ),
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="val_eye_eda.json",
        data_prefix=dict(img=""),
    )
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
<<<<<<< HEAD
        ann_file='test.json',
        data_prefix=dict(img='')
=======
        ann_file="test.json",
        data_prefix=dict(img=""),
>>>>>>> 67fd6022d8c5479b5129d05506284efe8328225e
    )
)

# Modify metric related settings
val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "val_eye_eda.json",
    metric="bbox",
    format_only=False,
    classwise=True,
)
test_evaluator = dict(ann_file=data_root + "test.json")

randomness = dict(seed=49)

# We can use the pre-trained Mask RCNN model to obtain higher performance
<<<<<<< HEAD
load_from = ''
=======
load_from = ""
>>>>>>> 67fd6022d8c5479b5129d05506284efe8328225e
