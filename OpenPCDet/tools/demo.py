import argparse, os
import glob
from pathlib import Path
import open3d as o3d
try:

    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        # data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list = glob.glob(str(root_path / '**' / f'*{self.ext}'), recursive=True) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list
        # self.sample_file_list = data_file_list[:100]

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        # 올바른 파일 형식에 따라 데이터 로드 방식을 결정합니다.
        if self.ext == '.bin':
            # .bin 파일은 binary 형식이므로 np.fromfile로 읽습니다.
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            # .npy 파일은 numpy 형식으로 저장된 데이터이므로 np.load로 읽습니다.
            points = np.load(self.sample_file_list[index])
        elif self.ext == '.pcd':
            # 수정된 부분: .pcd 파일은 ASCII 형식으로 처리합니다.
            points = self.load_pcd_file(self.sample_file_list[index])
        else:
            raise NotImplementedError(f"Unsupported file extension: {self.ext}")

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
###########################custum
    def load_pcd_file(self, pcd_file_path):
        """
        Load a .pcd file in ASCII format and parse it into a NumPy array.

        Args:
            pcd_file_path (str): Path to the .pcd file.

        Returns:
            np.ndarray: Array of shape (N, 4) containing x, y, z, intensity.
        """
        try:
            with open(pcd_file_path, 'r') as f:
                lines = f.readlines()

            # 헤더와 데이터 구분
            data_start_index = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("DATA"):
                    data_start_index = i + 1  # 데이터 섹션의 첫 번째 줄
                    break

            # 데이터 섹션 읽기
            points = np.loadtxt(lines[data_start_index:], dtype=np.float32)

            return points
        except Exception as e:
            raise RuntimeError(f"Error reading .pcd file: {pcd_file_path}, Error: {str(e)}")

# def box_to_points(box):
#     """
#     Convert a 3D bounding box to its corner points.
    
#     Args:
#         box (numpy.ndarray): Array of shape (7,) containing [x, y, z, dx, dy, dz, heading].
    
#     Returns:
#         numpy.ndarray: Array of shape (8, 3) containing the corner points of the box.
#     """
#     x, y, z, dx, dy, dz, heading = box
#     # Calculate half-dimensions
#     half_dx, half_dy, half_dz = dx / 2, dy / 2, dz / 2

#     # Define the corners of the box before rotation
#     corners = np.array([
#         [ half_dx,  half_dy, -half_dz],
#         [ half_dx, -half_dy, -half_dz],
#         [-half_dx, -half_dy, -half_dz],
#         [-half_dx,  half_dy, -half_dz],
#         [ half_dx,  half_dy,  half_dz],
#         [ half_dx, -half_dy,  half_dz],
#         [-half_dx, -half_dy,  half_dz],
#         [-half_dx,  half_dy,  half_dz],
#     ])

#     # Rotation matrix around z-axis
#     rotation_matrix = np.array([
#         [np.cos(heading), -np.sin(heading), 0],
#         [np.sin(heading),  np.cos(heading), 0],
#         [0,                0,               1]
#     ])

#     # Rotate and translate the corners
#     rotated_corners = np.dot(corners, rotation_matrix.T)
#     translated_corners = rotated_corners + np.array([x, y, z])

#     return translated_corners


def save_results(pred_boxes, pred_scores, pred_labels, save_path):
    """
    예측 결과를 텍스트 파일로 저장
    Args:
        pred_boxes: [N, 7] 형태의 numpy 배열 (x, y, z, dx, dy, dz, yaw)
        pred_scores: [N] 형태의 numpy 배열 (각 박스의 점수)
        pred_labels: [N] 형태의 numpy 배열 (각 박스의 클래스 레이블)
        save_path: 저장할 파일 경로
    """
    with open(save_path, 'w') as f:
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            # 속도는 없는 경우 0으로 설정
            vx, vy = 0.0, 0.0
            line = f"{box[0]:.6f} {box[1]:.6f} {box[2]:.6f} " \
                   f"{box[3]:.6f} {box[4]:.6f} {box[5]:.6f} {box[6]:.6f} " \
                   f"{vx:.6f} {vy:.6f} {score:.6f} {label}\n"
            f.write(line)
    print(f"Results saved to {save_path}")



###############################################
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--save_path', type=str, default="../output", help='specify the extension of your point cloud data file')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg





def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    os.makedirs(args.save_path, exist_ok=True)  # Ensure the save directory exists


    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # 결과 저장
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()

            sample_filename = os.path.splitext(os.path.basename(demo_dataset.sample_file_list[idx]))[0]
            save_file_path = os.path.join(args.save_path, f"{sample_filename}.txt")

            save_results(pred_boxes, pred_scores, pred_labels, save_file_path)
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )

            # # ▶ 수정된 부분: 시각화 함수 호출 방식 변경
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],save_path=args.save_path)

            # if not OPEN3D_FLAG:
            #     mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
