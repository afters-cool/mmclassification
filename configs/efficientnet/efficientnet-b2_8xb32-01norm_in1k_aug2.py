_base_ = [
    '../_base_/models/efficientnet_b2.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_conformer.py',
    '../_base_/default_runtime.py',
]

train_pipeline[1] = dict(type='RandomResizedCrop',size=260,efficientnet_style=True,interpolation='bicubic')
test_pipeline[1] = dict(type='CenterCrop',crop_size=260,efficientnet_style=True,interpolation='bicubic')

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))