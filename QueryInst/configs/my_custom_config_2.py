# the new config inherits the base configs to highlight the necessary modification
_base_ = './queryinst/queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py'

model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('box','back')
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='box/train_val/train.json',
        img_prefix='box/train_val/'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='box/train_val/valid.json',
        img_prefix='box/train_val/'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='box/test/ann_test_mini.json',
	#ann_file='box/test/ann_test_macro.json',
        img_prefix='box/test/'))

log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])


