import open3d as o3d
import os
import pandas as pd

file_paths = [
    "test_data/1727320102-53276943.pcd",
    "test_data/1727320101-961578277.pcd",
    "test_data/1727320102-53276943.pcd",
    "test_data/1727320102-153284974.pcd",
    "test_data/1727320102-253066301.pcd"
]

# file_paths = [os.path.join(pcd_directory, f) for f in pcd_files]

# 모든 PCD 파일 로드
point_clouds = []
pcd_info = []

for file_path in file_paths:
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        num_points = len(pcd.points)
        bounding_box = pcd.get_axis_aligned_bounding_box().get_extent() if num_points > 0 else None
        
        point_clouds.append(pcd)
        pcd_info.append({
            "File": file_path,
            "Num_Points": num_points,
            "Bounding_Box": bounding_box,
        })
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# PCD 정보 출력
pcd_info_df = pd.DataFrame(pcd_info)
print(pcd_info_df)

# PCD 파일 이어붙이기
combined_pcd = o3d.geometry.PointCloud()
for pcd in point_clouds:
    combined_pcd += pcd

# 결과 저장
output_file = "combined_test_data.pcd"
o3d.io.write_point_cloud(output_file, combined_pcd)
print(f"Combined PCD saved to {output_file}")

# 시각화
o3d.visualization.draw_geometries([combined_pcd], window_name="Combined PCD")



# import open3d as o3d
# import os
# import time

# # PCD 파일이 저장된 디렉토리 경로

# pcd_directory = "data/01_straight_walk"

# # 1초에 5프레임 = 0.2초 간격
# frame_interval = 0.2  # 초 단위

# # 모든 PCD 파일 경로 가져오기
# pcd_files = sorted([os.path.join(pcd_directory, f) for f in os.listdir(pcd_directory) if f.endswith(".pcd")])

# # 모든 PCD 파일을 순차적으로 읽어 이어붙임
# combined_pcd = o3d.geometry.PointCloud()
# for pcd_file in pcd_files:
#     # PCD 파일 읽기
#     pcd = o3d.io.read_point_cloud(pcd_file)
    
#     # 읽어온 PCD 데이터를 combined_pcd에 추가
#     combined_pcd += pcd
    
#     # 각 파일을 이어붙일 때 대기 시간 (0.2초 간격)
#     time.sleep(frame_interval)

# # 이어붙인 결과를 저장
# output_file = "data/combined_straight_walk.pcd"
# o3d.io.write_point_cloud(output_file, combined_pcd)
# print(f"Combined PCD saved to {output_file}")

# # 시각화
# o3d.visualization.draw_geometries([combined_pcd], window_name="Combined PCD")
