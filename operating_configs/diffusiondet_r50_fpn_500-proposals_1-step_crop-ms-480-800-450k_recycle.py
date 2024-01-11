# The new config inherits a base config to highlight the necessary modification
_base_ = 'projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=1), mask_head=dict(num_classes=1)))

custom_hooks = [
    dict(type='SubmissionHook', test_out_dir='submit')
]

# Modify dataset related settings
data_root = 'data/recycle/'
metainfo = {
    'classes': ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing',),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 230), (106, 0, 228), (60, 20, 220),
        (0, 80, 100), (0, 0, 70), (50, 0, 192), (250, 170, 30), (255, 0, 0)
    ]
}
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    bbox_head=dict(
        type='DynamicDiffusionDetHead',
        num_classes=10,
        feat_channels=256,
        num_proposals=500,
        num_heads=6,
        deep_supervision=True,
        prior_prob=0.01,
        snr_scale=2.0,
        sampling_timesteps=1,
        ddim_sampling_eta=1.0,
        single_head=dict(
            type='SingleDiffusionDetHead',
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            dropout=0.0,
            act_cfg=dict(type='ReLU', inplace=True),
            dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)),
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        # criterion
        criterion=dict(
            type='DiffusionDetCriterion',
            num_classes=10,
            assigner=dict(
                type='DiffusionDetMatcher',
                match_costs=[
                    dict(
                        type='FocalLossCost',
                        alpha=0.25,
                        gamma=2.0,
                        weight=2.0,
                        eps=1e-8),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)
                ],
                center_radius=2.5,
                candidate_topk=5),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                alpha=0.25,
                gamma=2.0,
                reduction='sum',
                loss_weight=2.0),
            loss_bbox=dict(type='L1Loss', reduction='sum', loss_weight=5.0),
            loss_giou=dict(type='GIoULoss', reduction='sum',
                           loss_weight=2.0))
    ))

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.json',
        data_prefix=dict(img='')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.json',
        data_prefix=dict(img='')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test.json',
        data_prefix=dict(img='')))

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'train.json')
test_evaluator = dict(ann_file=data_root + 'test.json')

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/diffusiondet/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco_20230215_090925-7d6ed504.pth'