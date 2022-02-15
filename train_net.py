#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer import UBTeacherTrainer, BaselineTrainer

# hacky way to register
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import ubteacher.data.datasets.builtin

from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel

# register customized datasets
from detectron2.data.datasets import register_coco_instances


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # img_dir = "/home/ims_dev_ml/finetune_gptj6b/pytago-ml/pytago/object_detection/datasets/wheat-detection/annotations/sup10_annotations.json"
    # ann_path = "/home/ims_dev_ml/finetune_gptj6b/pytago-ml/pytago/object_detection/datasets/wheat-detection/train2017"
    # cfg.IMAGE_DIR = img_dir
    # cfg.ANN_PATH = ann_path
    # cfg.DATASETS.NAME = "wheat-detection"
    
    cfg.DATASETS.TRAIN = (cfg.DATASETS.NAME+"_train", )
    register_coco_instances(cfg.DATASETS.TRAIN[0], {}, cfg.DATASETS.TRAIN_ANN_PATH, cfg.DATASETS.TRAIN_IMAGE_DIR)
    # cfg.DATASETS.TEST = (cfg.DATASETS.NAME+"_val", )
    # register_coco_instances(cfg.DATASETS.TEST[0], {}, cfg.DATASETS.TEST_ANN_PATH, cfg.DATASETS.TEST_IMAGE_DIR)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    print(f"Model will save at {cfg.OUTPUT_DIR}")

    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ubteacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )