_base_ = [
    '../_base_/models/efficientnet_b2.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_conformer.py',
    '../_base_/default_runtime.py',
]

# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=260,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    dict(
        type='Albu', transforms=[dict(type='ChannelShuffle', p=1)])
    # dict(
    #     type='Albu',
    #     transforms=[
    #         # dict(type='Affine', rotate=(-15, 15), translate_percent=(0.0, 0.25), shear=(-3, 3), p=0.5),    
    #         # dict(type='ToGray', p=0.1),
    #         # dict(type='GaussianBlur', blur_limit=(3, 7), p=0.05),
    #         # dict(type='GaussNoise', p=0.05),
    #         # dict(type='RandomGridShuffle', grid=(2, 2), p=0.3),
    #         # dict(type='Posterize', p=0.2),
    #         # dict(type='RandomBrightnessContrast', p=0.5),
    #         # dict(type='Cutout', p=0.05),
    #         # dict(type='RandomSnow', p=0.1),
    #         # dict(
    #         #     type='ShiftScaleRotate',
    #         #     shift_limit=0.0625,
    #         #     scale_limit=0.0,
    #         #     rotate_limit=0,
    #         #     interpolation=1,
    #         #     p=0.5),
    #         # dict(
    #         #     type='RandomBrightnessContrast',
    #         #     brightness_limit=[0.1, 0.3],
    #         #     contrast_limit=[0.1, 0.3],
    #         #     p=0.2),
    #         # dict(type='ChannelShuffle', p=0.1),
    #         # dict(
    #         #     type='OneOf',
    #         #     transforms=[
    #         #         dict(type='Blur', blur_limit=3, p=1.0),
    #         #         dict(type='MedianBlur', blur_limit=3, p=1.0)
    #         #     ],
    #         #     p=0.1),
    #     ]),
    
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CenterCrop',
        crop_size=260,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
