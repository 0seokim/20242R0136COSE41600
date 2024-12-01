import argparse
import torch
import numpy as np
from pathlib import Path

from OpenPCDet.pcdet.config import cfg, cfg_from_yaml_file
from OpenPCDet.pcdet.datasets import build_dataloader
from OpenPCDet.pcdet.models import build_network
from OpenPCDet.pcdet.utils import common_utils
from OpenPCDet.tools.eval_utils import eval_utils

import open3d as o3d
import numpy as np

def parse_config():
    parser = argparse.ArgumentParser(description="Point Cloud Detection Inference")
    parser.add_argument("--cfg_file", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--pcd_file", type=str, required=True, help="Path to input PCD file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for dataloader")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--save_to_file", action="store_true", help="Save inference results to file")
    parser.add_argument("--output_file", type=str, default="pedestrian_results.txt", help="Path to save the results")
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])  # remove cfgs/ and filename

    return args, cfg


# def load_pcd_file(pcd_file_path):
#     import open3d as o3d
#     points = np.load(pcd_file_path)
#     # pcd = o3d.io.read_point_cloud(pcd_file_path)
#     # points = np.asarray(pcd.points, dtype=np.float32)
#     return points

def load_pcd_file(pcd_file_path):
    # .pcd 파일 읽기
    pcd = o3d.io.read_point_cloud(pcd_file_path, format="pcd")
    if hasattr(pcd, 'points') and hasattr(pcd, 'colors'):  # xyz + intensity일 가능성 확인
        points = np.asarray(pcd.points, dtype=np.float32)
        if hasattr(pcd, 'intensities'):  # intensity가 Open3D 내장 지원될 때
            intensities = np.asarray(pcd.intensities, dtype=np.float32)
            return np.hstack((points, intensities.reshape(-1, 1)))
    return points


def main():
    args, cfg = parse_config()

    logger = common_utils.create_logger()
    logger.info("**********************Start logging**********************")

    class_names = cfg.CLASS_NAMES

    # Dataset and Dataloader
    demo_dataset = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=class_names,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        training=False,
        logger=logger,
    )[0]  # First element is dataset

    # Load Model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(class_names), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.eval()

    # Load PCD
    points = load_pcd_file(args.pcd_file)
    points = torch.from_numpy(points).float()

    pedestrian_results = []  # To store pedestrian detection results

    # Inference
    with torch.no_grad():
        data_dict = {
            "points": points,
            "frame_id": 0,
        }
        data_dict = demo_dataset.prepare_data(data_dict=data_dict)
        pred_dicts, _ = model.forward(data_dict)

    # Process Results
    for pred_dict in pred_dicts:
        pred_boxes = pred_dict["pred_boxes"].cpu().numpy()
        pred_scores = pred_dict["pred_scores"].cpu().numpy()
        pred_labels = pred_dict["pred_labels"].cpu().numpy()

        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            if class_names[label - 1] == "Pedestrian":
                result = f"Pedestrian Detected: Box: {box.tolist()}, Score: {score}"
                logger.info(result)
                pedestrian_results.append(result)

    # Save Results to File
    if args.save_to_file:
        with open(args.output_file, "w") as f:
            f.write("\n".join(pedestrian_results))
        logger.info(f"Pedestrian detection results saved to {args.output_file}")

    logger.info("**********************End logging**********************")


if __name__ == "__main__":
    main()
