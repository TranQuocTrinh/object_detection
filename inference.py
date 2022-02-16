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

from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel

import cv2
import PIL
import os
import torch
import numpy as np
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json


def setup(model_dir):
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(os.path.join(model_dir, "config.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_final.pth")
    
    # register_dataset
    cfg.DATASETS.TRAIN = (cfg.DATASETS.NAME+"_train", )
    register_coco_instances(cfg.DATASETS.TRAIN[0], {}, cfg.DATASETS.TRAIN_ANN_PATH, cfg.DATASETS.TRAIN_IMAGE_DIR)
    # cfg.DATASETS.TEST = (cfg.DATASETS.NAME+"_val", )
    # register_coco_instances(cfg.DATASETS.TEST[0], {}, cfg.DATASETS.TEST_ANN_PATH, cfg.DATASETS.TEST_IMAGE_DIR)

    load_coco_json(cfg.DATASETS.TRAIN_ANN_PATH, cfg.DATASETS.TRAIN_IMAGE_DIR, cfg.DATASETS.TRAIN[0])

    return cfg


def build_model(cfg):
    Trainer = UBTeacherTrainer
    model = Trainer.build_model(cfg)
    model_teacher = Trainer.build_model(cfg)
    ensem_ts_model = EnsembleTSModel(model_teacher, model)
    DetectionCheckpointer(ensem_ts_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume="resume")
    ensem_ts_model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS)["model"])

    model = ensem_ts_model.modelTeacher
    model.eval()
    return model


def inference(cfg, model, image_path, threshold=0.7, visualize=None): #visualize is path output of image visualize
    # image = cv2.imread(image_path)
    pil_image = PIL.Image.open(image_path).convert('RGB') 
    image = np.array(pil_image)
    image = image[:, :, ::-1].copy()

    inputs = torch.from_numpy(np.transpose(image, (2, 0, 1)))
    outputs = model([{"image":inputs}])
    instances = outputs[0]["instances"].to("cpu")
    instances = instances[instances.scores > threshold]
    
    if visualize is not None:
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
        out = v.draw_instance_predictions(instances)
        cv2.imwrite(visualize, out.get_image()[:, :, ::-1])

    return instances

    """
    Manual way

    height, width = image.shape[:2]
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": height, "width": width}]
    with torch.no_grad():
        images = model.preprocess_image(inputs)  # don't forget to preprocess
        features = model.backbone(images.tensor)  # set of cnn features
        proposals, _ = model.proposal_generator(images, features, None)  # RPN

        features_ = [features[f] for f in model.roi_heads.box_in_features]
        box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates
        predictions = model.roi_heads.box_predictor(box_features)
        pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
        pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)

        # output boxes, masks, scores, etc
        pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
        # features of the proposed boxes
        feats = box_features[pred_inds]
    """

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="visulize_test/", type=str,required=True, help="Directory output visualize test",)
    parser.add_argument("--test_dir", default="datasets/dota/test/images/", type=str,required=True, help="Directory test images",)
    parser.add_argument("--model_dir", default="output/", type=str,required=True, help="Directory model",)
    parser.add_argument("--threshold", default=0.8, type=float,required=False, help="Threshold",)
    args = parser.parse_args()

    cfg = setup(args.model_dir)
    model = build_model(cfg)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    files = [args.test_dir + f for f in os.listdir(args.test_dir) if f.split(".")[-1].lower() in {"jpg", "png", "jpeg"}]
    import random
    from tqdm import tqdm
    # files = random.choices(files, k=30)
    for idx in tqdm(range(len(files))):
        # idx = int(input("Input index image: "))
        # if idx == -1:
        #     break
        # print(f"image_path: {files[idx]}")
        output_file = args.output_dir + ".".join(files[idx].replace(args.test_dir, "").split(".")[:-1]) + ".jpg"
        result = inference(cfg, model=model, image_path=files[idx], threshold=args.threshold, visualize=output_file)
        # print(result)

if __name__ == "__main__":
    main()
