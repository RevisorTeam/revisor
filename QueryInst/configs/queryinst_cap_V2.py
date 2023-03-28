# the new config inherits the base configs to highlight the necessary modification
_base_ = './queryinst/queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py'

model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

# 1. dataset settings
dataset_type = 'CocoDataset'
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])


