import torch
import numpy as np

from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh, bbox_flip
from mmdet.core.bbox.samplers import PseudoSampler
from ..builder import HEADS
from .cascade_roi_head import CascadeRoIHead

from mmcv.ops.nms import batched_nms

def mask2results(mask_preds, det_labels, num_classes):
    cls_segms = [[] for _ in range(num_classes)]
    for i in range(mask_preds.shape[0]):
        cls_segms[det_labels[i]].append(mask_preds[i])
    return cls_segms

@HEADS.register_module()
class QueryRoIHead(CascadeRoIHead):
    r"""

    Args:
        num_stages (int): Number of stage whole iterative process.
            Defaults to 6.
        stage_loss_weights (Tuple[float]): The loss
            weight of each stage. By default all stages have
            the same weight 1.
        bbox_roi_extractor (dict): Config of box roi extractor.
        bbox_head (dict): Config of box head.
        train_cfg (dict, optional): Configuration information in train stage.
            Defaults to None.
        test_cfg (dict, optional): Configuration information in test stage.
            Defaults to None.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    """

    def __init__(self,
                 num_stages=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 proposal_feature_channel=256,
                 bbox_roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=7, sampling_ratio=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 mask_roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=14, sampling_ratio=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 bbox_head=dict(
                     type='DIIHead',
                     num_classes=80,
                     num_fcs=2,
                     num_heads=8,
                     num_cls_fcs=1,
                     num_reg_fcs=3,
                     feedforward_channels=2048,
                     hidden_channels=256,
                     dropout=0.0,
                     roi_feat_size=7,
                     ffn_act_cfg=dict(type='ReLU', inplace=True)),
                 mask_head=dict(
                     type='DynamicMaskHead',
                     dynamic_conv_cfg=dict(
                         type='DynamicConv',
                         in_channels=256,
                         feat_channels=64,
                         out_channels=256,
                         input_feat_shape=14,
                         with_proj=False,
                         act_cfg=dict(type='ReLU', inplace=True),
                         norm_cfg=dict(type='LN')),
                     dropout=0.0,
                     num_convs=4,
                     roi_feat_size=14,
                     in_channels=256,
                     conv_kernel_size=3,
                     conv_out_channels=256,
                     class_agnostic=False,
                     norm_cfg=dict(type='BN'),
                     upsample_cfg=dict(type='deconv', scale_factor=2),
                     loss_dice=dict(type='DiceLoss', loss_weight=8.0)),
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        assert bbox_roi_extractor is not None
        assert mask_roi_extractor is not None
        assert bbox_head is not None
        assert mask_head is not None
        assert len(stage_loss_weights) == num_stages
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.proposal_feature_channel = proposal_feature_channel
        super(QueryRoIHead, self).__init__(
            num_stages,
            stage_loss_weights,
            bbox_roi_extractor=bbox_roi_extractor,
            mask_roi_extractor=mask_roi_extractor,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(self.bbox_sampler[stage], PseudoSampler), \
                    'QueryInst only support `PseudoSampler`'

    def _bbox_forward(self, stage, x, rois, object_feats, img_metas):
        """Box head forward function used in both training and testing. Returns
        all regression, classification results and a intermediate feature.

        Args:
            stage (int): The index of current stage in
                iterative process.
            x (List[Tensor]): List of FPN features
            rois (Tensor): Rois in total batch. With shape (num_proposal, 5).
                the last dimension 5 represents (img_index, x1, y1, x2, y2).
            object_feats (Tensor): The object feature extracted from
                the previous stage.
            img_metas (dict): meta information of images.

        Returns:
            dict[str, Tensor]: a dictionary of bbox head outputs,
                Containing the following results:

                    - cls_score (Tensor): The score of each class, has
                      shape (batch_size, num_proposals, num_classes)
                      when use focal loss or
                      (batch_size, num_proposals, num_classes+1)
                      otherwise.
                    - decode_bbox_pred (Tensor): The regression results
                      with shape (batch_size, num_proposal, 4).
                      The last dimension 4 represents
                      [tl_x, tl_y, br_x, br_y].
                    - object_feats (Tensor): The object feature extracted
                      from current stage
                    - detach_cls_score_list (list[Tensor]): The detached
                      classification results, length is batch_size, and
                      each tensor has shape (num_proposal, num_classes).
                    - detach_proposal_list (list[tensor]): The detached
                      regression results, length is batch_size, and each
                      tensor has shape (num_proposal, 4). The last
                      dimension 4 represents [tl_x, tl_y, br_x, br_y].
        """
        num_imgs = len(img_metas)
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        cls_score, bbox_pred, object_feats, attn_feats = bbox_head(bbox_feats,
                                                       object_feats)
        proposal_list = self.bbox_head[stage].refine_bboxes(
            rois,
            rois.new_zeros(len(rois)),  # dummy arg
            bbox_pred.view(-1, bbox_pred.size(-1)),
            [rois.new_zeros(object_feats.size(1)) for _ in range(num_imgs)],
            img_metas)
        bbox_results = dict(
            cls_score=cls_score,
            decode_bbox_pred=torch.cat(proposal_list),
            object_feats=object_feats,
            attn_feats=attn_feats,
            # detach then use it in label assign
            detach_cls_score_list=[
                cls_score[i].detach() for i in range(num_imgs)
            ],
            detach_proposal_list=[item.detach() for item in proposal_list])

        return bbox_results

    def _mask_forward(self, stage, x, rois, attn_feats):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        mask_pred = mask_head(mask_feats, attn_feats)

        mask_results = dict(mask_pred=mask_pred)
        return mask_results

    def _mask_forward_train(self, stage, x, attn_feats, sampling_results, gt_masks, rcnn_train_cfg):

        if sum([len(gt_mask) for gt_mask in gt_masks])==0:
            print('Ground Truth Not Found!')
            loss_mask = sum([_.sum() for _ in self.mask_head[stage].parameters()]) * 0.
            return dict(loss_mask=loss_mask)
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        attn_feats = torch.cat([feats[res.pos_inds] for (feats, res) in zip(attn_feats, sampling_results)])
        mask_results = self._mask_forward(stage, x, pos_rois, attn_feats)

        mask_targets = self.mask_head[stage].get_targets(
            sampling_results, gt_masks, rcnn_train_cfg)

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'],
                                               mask_targets, pos_labels)
        mask_results.update(loss_mask)
        return mask_results

    def forward_train(self,
                      x,
                      proposal_boxes,
                      proposal_features,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_masks=None):
        """Forward function in training stage.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposals (Tensor): Decoded proposal bboxes, has shape
                (batch_size, num_proposals, 4)
            proposal_features (Tensor): Expanded proposal
                features, has shape
                (batch_size, num_proposals, proposal_feature_channel)
            img_metas (list[dict]): list of image docs dict where
                each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape',
                'pad_shape', and 'img_norm_cfg'. For details on the
                values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            imgs_whwh (Tensor): Tensor with shape (batch_size, 4),
                    the dimension means
                    [img_width,img_height, img_width, img_height].
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components of all stage.
        """

        num_imgs = len(img_metas)
        num_proposals = proposal_boxes.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)
        all_stage_bbox_results = []
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        object_feats = proposal_features
        all_stage_loss = {}
        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                              img_metas)
            all_stage_bbox_results.append(bbox_results)
            if gt_bboxes_ignore is None:
                # TODO support ignore
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            cls_pred_list = bbox_results['detach_cls_score_list']
            proposal_list = bbox_results['detach_proposal_list']
            for i in range(num_imgs):
                normolize_bbox_ccwh = bbox_xyxy_to_cxcywh(proposal_list[i] /
                                                          imgs_whwh[i])
                assign_result = self.bbox_assigner[stage].assign(
                    normolize_bbox_ccwh, cls_pred_list[i], gt_bboxes[i],
                    gt_labels[i], img_metas[i])
                sampling_result = self.bbox_sampler[stage].sample(
                    assign_result, proposal_list[i], gt_bboxes[i])
                sampling_results.append(sampling_result)
            bbox_targets = self.bbox_head[stage].get_targets(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg[stage],
                True)
            cls_score = bbox_results['cls_score']
            decode_bbox_pred = bbox_results['decode_bbox_pred']
            object_feats = bbox_results['object_feats']

            single_stage_loss = self.bbox_head[stage].loss(
                cls_score.view(-1, cls_score.size(-1)),
                decode_bbox_pred.view(-1, 4),
                *bbox_targets,
                imgs_whwh=imgs_whwh)

            if self.with_mask:
                mask_results = self._mask_forward_train(stage, x, bbox_results['attn_feats'], 
                                            sampling_results, gt_masks, self.train_cfg[stage])
                single_stage_loss['loss_mask'] = mask_results['loss_mask']

            for key, value in single_stage_loss.items():
                all_stage_loss[f'stage{stage}_{key}'] = value * \
                                    self.stage_loss_weights[stage]

        return all_stage_loss

    def simple_test(self,
                    x,
                    proposal_boxes,
                    proposal_features,
                    img_metas,
                    imgs_whwh,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_boxes (Tensor): Decoded proposal bboxes, has shape
                (batch_size, num_proposals, 4)
            proposal_features (Tensor): Expanded proposal
                features, has shape
                (batch_size, num_proposals, proposal_feature_channel)
            img_metas (dict): meta information of images.
            imgs_whwh (Tensor): Tensor with shape (batch_size, 4),
                    the dimension means
                    [img_width,img_height, img_width, img_height].
            rescale (bool): If True, return boxes in original image
                space. Defaults to False.

        Returns:
            bbox_results (list[tuple[np.ndarray]]): \
                [[cls1_det, cls2_det, ...], ...]. \
                The outer list indicates images, and the inner \
                list indicates per-class detected bboxes. The \
                np.ndarray has shape (num_det, 5) and the last \
                dimension 5 represents (x1, y1, x2, y2, score).
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        # Decode initial proposals
        num_imgs = len(img_metas)
        proposal_list = [proposal_boxes[i] for i in range(num_imgs)]
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}

        object_feats = proposal_features
        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                              img_metas)
            object_feats = bbox_results['object_feats']
            cls_score = bbox_results['cls_score']
            proposal_list = bbox_results['detach_proposal_list']

        if self.with_mask:
            rois = bbox2roi(proposal_list)
            mask_results = self._mask_forward(stage, x, rois, bbox_results['attn_feats'])
            mask_results['mask_pred'] = mask_results['mask_pred'].reshape(
                num_imgs, -1, *mask_results['mask_pred'].size()[1:]
            )

        num_classes = self.bbox_head[-1].num_classes
        det_bboxes = []
        det_labels = []

        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        for img_id in range(num_imgs):
            cls_score_per_img = cls_score[img_id]
            scores_per_img, topk_indices = cls_score_per_img.flatten(
                0, 1).topk(
                    self.test_cfg.max_per_img, sorted=False)
            labels_per_img = topk_indices % num_classes
            bbox_pred_per_img = proposal_list[img_id][topk_indices //
                                                      num_classes]
            if rescale:
                scale_factor = img_metas[img_id]['scale_factor']
                bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
            det_bboxes.append(
                torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1))
            det_labels.append(labels_per_img)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
            _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
            segm_results = []
            mask_pred = mask_results['mask_pred']
            for img_id in range(num_imgs):
                mask_pred_per_img = mask_pred[img_id].flatten(0, 1)[topk_indices]
                mask_pred_per_img = mask_pred_per_img[:, None, ...].repeat(1, num_classes, 1, 1)
                segm_result = self.mask_head[-1].get_seg_masks(
                    mask_pred_per_img, _bboxes[img_id], det_labels[img_id],
                    self.test_cfg, ori_shapes[img_id], scale_factors[img_id],
                    rescale)
                segm_results.append(segm_result)

            ms_segm_result['ensemble'] = segm_results
        
        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results

    def aug_test(self,
                 aug_x,
                 aug_proposal_boxes,
                 aug_proposal_features,
                 aug_img_metas,
                 aug_imgs_whwh,
                 rescale=False):
        
        samples_per_gpu = len(aug_img_metas[0])
        aug_det_bboxes = [[] for _ in range(samples_per_gpu)]
        aug_det_labels = [[] for _ in range(samples_per_gpu)]
        aug_mask_preds = [[] for _ in range(samples_per_gpu)]
        for x, proposal_boxes, proposal_features, img_metas, imgs_whwh in \
            zip(aug_x, aug_proposal_boxes, aug_proposal_features, aug_img_metas, aug_imgs_whwh):
            

            num_imgs = len(img_metas)
            proposal_list = [proposal_boxes[i] for i in range(num_imgs)]
            ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
            scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

            object_feats = proposal_features
            for stage in range(self.num_stages):
                rois = bbox2roi(proposal_list)
                bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                                img_metas)
                object_feats = bbox_results['object_feats']
                cls_score = bbox_results['cls_score']
                proposal_list = bbox_results['detach_proposal_list']
            
            if self.with_mask:
                rois = bbox2roi(proposal_list)
                mask_results = self._mask_forward(stage, x, rois, bbox_results['attn_feats'])
                mask_results['mask_pred'] = mask_results['mask_pred'].reshape(
                    num_imgs, -1, *mask_results['mask_pred'].size()[1:]
                )
            
            num_classes = self.bbox_head[-1].num_classes
            det_bboxes = []
            det_labels = []

            if self.bbox_head[-1].loss_cls.use_sigmoid:
                cls_score = cls_score.sigmoid()
            else:
                cls_score = cls_score.softmax(-1)[..., :-1]

            for img_id in range(num_imgs):
                cls_score_per_img = cls_score[img_id]
                scores_per_img, topk_indices = cls_score_per_img.flatten(
                    0, 1).topk(
                        self.test_cfg.max_per_img, sorted=False)
                labels_per_img = topk_indices % num_classes
                bbox_pred_per_img = proposal_list[img_id][topk_indices //
                                                        num_classes]
                if rescale:
                    scale_factor = img_metas[img_id]['scale_factor']
                    bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
                aug_det_bboxes[img_id].append(
                    torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1))
                det_bboxes.append(
                    torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1))
                aug_det_labels[img_id].append(labels_per_img)
                det_labels.append(labels_per_img)
            
            if self.with_mask:
                if rescale and not isinstance(scale_factors[0], float):
                        scale_factors = [
                            torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                            for scale_factor in scale_factors
                        ]
                _bboxes = [
                        det_bboxes[i][:, :4] *
                        scale_factors[i] if rescale else det_bboxes[i][:, :4]
                        for i in range(len(det_bboxes))
                    ]
                mask_pred = mask_results['mask_pred']
                for img_id in range(num_imgs):
                    mask_pred_per_img = mask_pred[img_id].flatten(0, 1)[topk_indices]
                    mask_pred_per_img = mask_pred_per_img[:, None, ...].repeat(1, num_classes, 1, 1)
                    segm_result = self.mask_head[-1].get_seg_masks(
                        mask_pred_per_img, _bboxes[img_id], det_labels[img_id],
                        self.test_cfg, ori_shapes[img_id], scale_factors[img_id],
                        rescale, format=False)
                    aug_mask_preds[img_id].append(segm_result.detach().cpu().numpy())

        det_bboxes, det_labels, mask_preds = [], [], []

        for img_id in range(samples_per_gpu):
            for aug_id in range(len(aug_det_bboxes[img_id])):
                img_meta = aug_img_metas[aug_id][img_id]
                img_shape = img_meta['ori_shape']
                flip = img_meta['flip']
                flip_direction = img_meta['flip_direction']
                aug_det_bboxes[img_id][aug_id][:, :-1] = bbox_flip(aug_det_bboxes[img_id][aug_id][:, :-1],
                                                    img_shape, flip_direction) if flip else aug_det_bboxes[img_id][aug_id][:, :-1]
                if flip:
                    if flip_direction == 'horizontal':
                        aug_mask_preds[img_id][aug_id] = aug_mask_preds[img_id][aug_id][:, :, ::-1]
                    else:
                        aug_mask_preds[img_id][aug_id] = aug_mask_preds[img_id][aug_id][:, ::-1, :]

        for img_id in range(samples_per_gpu):
            det_bboxes_per_im = torch.cat(aug_det_bboxes[img_id])
            det_labels_per_im = torch.cat(aug_det_labels[img_id])
            mask_preds_per_im = np.concatenate(aug_mask_preds[img_id])

            # TODO(vealocia): implement batched_nms here.
            det_bboxes_per_im, keep_inds = batched_nms(det_bboxes_per_im[:, :-1], det_bboxes_per_im[:, -1].contiguous(), det_labels_per_im, self.test_cfg.nms)
            det_bboxes_per_im = det_bboxes_per_im[:self.test_cfg.max_per_img, ...]
            det_labels_per_im = det_labels_per_im[keep_inds][:self.test_cfg.max_per_img, ...]
            mask_preds_per_im = mask_preds_per_im[keep_inds.detach().cpu().numpy()][:self.test_cfg.max_per_img, ...]
            det_bboxes.append(det_bboxes_per_im)
            det_labels.append(det_labels_per_im)
            mask_preds.append(mask_preds_per_im)
    
        ms_bbox_result = {}
        ms_segm_result = {}
        num_classes = self.bbox_head[-1].num_classes
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], num_classes)
            for i in range(samples_per_gpu)
        ]
        ms_bbox_result['ensemble'] = bbox_results
        mask_results = [
            mask2results(mask_preds[i], det_labels[i], num_classes)
            for i in range(samples_per_gpu)
        ]
        ms_segm_result['ensemble'] = mask_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']
        return results

    def forward_dummy(self, x, proposal_boxes, proposal_features, img_metas):
        """Dummy forward function when do the flops computing."""
        all_stage_bbox_results = []
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        object_feats = proposal_features
        if self.with_bbox:
            for stage in range(self.num_stages):
                rois = bbox2roi(proposal_list)
                bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                                  img_metas)

                all_stage_bbox_results.append(bbox_results)
                proposal_list = bbox_results['detach_proposal_list']
                object_feats = bbox_results['object_feats']
        return all_stage_bbox_results
