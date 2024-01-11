# The new config inherits a base config to highlight the necessary modification
_base_ = '../mmdetection/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco.py'

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

# for nomalize cfg
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# test 3 Albu
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.4],
        contrast_limit=[0.1, 0.2],
        p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

RANDAUG_SPACE = [
    [dict(type='AutoContrast')], [dict(type='Rotate')],
    [dict(type='Posterize')], [dict(type='Solarize')],
    [dict(type='Brightness')], [dict(type='Sharpness')]
]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),

    # 1 RandomchoiceResize 
    dict(type='RandomChoiceResize',
                scales=[(480, 1333), (512, 1333), (544, 1333),
                (576, 1333), (608, 1333), (640, 1333),
                (672, 1333), (704, 1333), (736, 1333),
                (768, 1333), (800, 1333)], 
                keep_ratio=True),

    # test 2 RandAug + AutoAug
    # RandAugment : In RandAug_space, the required augmentation techniques are defined, 
    # and a random selection of aug_num techniques is applied.
    dict(type='RandAugment', aug_space=RANDAUG_SPACE, aug_num=2),  

    # AutoAugment defines and applies augmentation policies, 
    # which are sets of specific augmentation operations 
    # with associated probabilities and magnitudes.
    dict(type='AutoAugment', # 정책이 두가지. 중에 랜덤으로 들어감. 
        policies=[[dict(type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
                ],
                [dict(type='Resize',
                img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                multiscale_mode='value',
                keep_ratio=True),
                dict(type='RandomCrop',
                crop_type='absolute_range',
                crop_size=(384, 600),
                allow_negative_crop=True),
                dict(type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333),
                           (576, 1333), (608, 1333), (640, 1333),
                           (672, 1333), (704, 1333), (736, 1333),
                           (768, 1333), (800, 1333)],
                multiscale_mode='value',
                override=True,
                keep_ratio=True)
                ]
                ]),  

    # test 3 Albu
    dict(type='Albu', transforms=albu_train_transforms),

    # test 4 Mosaic + RandomAffine
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        bbox_clip_border=True),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        bbox_clip_border=True),

    # test 5 PhotoMetricDistortion
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),

    # test 6 CopyPaste
    dict(type='CopyPaste', max_num_pasted=100),
    # #normalize? 
    # dict(type='Normalize', **img_norm_cfg),

    # dict(
    #     type='RandomChoiceResize',
    #     scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
    #             (1333, 768), (1333, 800)],
    #     keep_ratio=True),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs') # meta info 들 맞춰주는 
]

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_eye_eda.json', #train.json
        data_prefix=dict(img='')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val_eye_eda.json', #train.json
        data_prefix=dict(img='')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test.json',
        data_prefix=dict(img='')))

# hooks
custom_hooks = [
    dict(type='SubmissionHook', test_out_dir='submit')
]
# default_hooks= dict(
#     visualization=dict(type='DetVisualizationHook',
#     draw=True,
#     interval=1,
#     show=True))

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'val_eye_eda.json') #train.json
test_evaluator = dict(ann_file=data_root + 'test.json')

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'