import open3d as o3d
import numpy as np
import os

def generate_preprocessed_path(file_path):
    # 파일의 디렉토리, 이름, 확장자 분리
    directory, filename = os.path.split(file_path)
    name, ext = os.path.splitext(filename)
    
    # 새로운 파일 이름 생성 (preprocessed 추가)
    new_filename = f"{name}_preprocessed{ext}"
    
    # 새 경로 생성
    new_path = os.path.join(directory, new_filename)
    return new_path

# --- 1. 개별 PCD 파일 전처리 ---
def preprocess_pcd(file_path):
    # PCD 파일 읽기
    pcd = o3d.io.read_point_cloud(file_path)

    # Voxel Downsampling
    downsample_pcd = pcd.voxel_down_sample(voxel_size=0.01)
    
    # Outlier 제거
    pcd, ind = downsample_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    ror_pcd = downsample_pcd.select_by_index(ind)


    # RANSAC을 사용하여 평면 추정
    plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                                ransac_n=3,
                                                num_iterations=2000)

    # 도로에 속하지 않는 포인트 (outliers) 추출
    final_point = ror_pcd.select_by_index(inliers, invert=True)

    # Pass-through 필터링 (Z축 범위 지정)
    points = np.asarray(final_point.points)

     # 필터링 전 Z축 범위 확인
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    print(f"Before Pass-through Filtering: Z Min = {z_min}, Z Max = {z_max}")
    
    # Z축 범위 필터링
    filtered_points = points[(points[:, 2] < 4)]  # 0 < Z < 5
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    # 필터링 후 Z축 범위 확인
    points_after = np.asarray(pcd.points)
    z_min_after, z_max_after = np.min(points_after[:, 2]), np.max(points_after[:, 2])
    print(f"After Pass-through Filtering: Z Min = {z_min_after}, Z Max = {z_max_after}")

    
    # 좌표계 변환 (원점 이동)
    center = np.mean(np.asarray(pcd.points), axis=0)
    pcd.translate(-center)
    
    # 결과 반환
    return pcd

# pcd 파일 불러오고 시각화하는 함수
def load_and_visualize_pcd(pcd, point_size=1.0):
    # pcd 파일 로드
    print(f"Point cloud has {len(pcd.points)} points.")
    
    # 시각화 설정
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()


# PCD 파일 불러오기 및 데이터 확인 함수
def load_and_inspect_pcd(pcd):
    # PCD 파일 로드
    
    # 점 구름 데이터를 numpy 배열로 변환
    points = np.asarray(pcd.points)
    
    # 점 데이터 개수 및 일부 점 확인
    print(f"Number of points: {len(points)}")
    print("First 5 points:")
    print(points[:5])  # 처음 5개의 점 출력
    
    # 점의 x, y, z 좌표의 범위 확인
    print("X coordinate range:", np.min(points[:, 0]), "to", np.max(points[:, 0]))
    print("Y coordinate range:", np.min(points[:, 1]), "to", np.max(points[:, 1]))
    print("Z coordinate range:", np.min(points[:, 2]), "to", np.max(points[:, 2]))

file_path = "test_data/1727320101-665925967.pcd"
# pcd 시각화 테스트

preprocessed_pcd = preprocess_pcd(file_path)
# load_and_visualize_pcd(preprocessed_pcd, 0.5)
# load_and_inspect_pcd(preprocessed_pcd)

output_file = generate_preprocessed_path(file_path)
o3d.io.write_point_cloud(output_file, preprocessed_pcd)
print(f"Combined PCD saved to {output_file}")
