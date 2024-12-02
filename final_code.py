# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# pcd 파일 불러오기, 필요에 맞게 경로 수정
# file_path = "test_data/1727320101-665925967.pcd"
# output_image_path = "test_data/captured_image.png"
# pcd_folder = "test_data"
# output_folder = "test_data/captured_images"
# output_video_path = "test_data/output_video.mp4"
# 
# pcd_folder = "todo_data/01_straight_walk/pcd"  # PCD 파일이 저장된 폴더
# output_folder = "todo_data/01_straight_walk/captured_images"  # 이미지를 저장할 폴더
# video_output_path = "todo_data/01_straight_walk/output_video.mp4"  # 생성될 영상 경로
pcd_folder = "todo_data/05_straight_duck_walk/pcd"  # PCD 파일이 저장된 폴더
output_folder = "todo_data/05_straight_duck_walk/captured_images"  # 이미지를 저장할 폴더
video_output_path = "todo_data/05_straight_duck_walk/output_video.mp4" # 생성될 영상 경로
# pcd_folder = "todo_data/02_straight_duck_walk/pcd"  # PCD 파일이 저장된 폴더
# output_folder = "todo_data/02_straight_duck_walk/captured_images"  # 이미지를 저장할 폴더
# video_output_path = "todo_data/02_straight_duck_walk/output_video.mp4"  # 생성될 영상 경로
# pcd_folder = "todo_data/03_straight_crawl/pcd"  # PCD 파일이 저장된 폴더
# output_folder = "todo_data/03_straight_crawl/captured_images"  # 이미지를 저장할 폴더
# video_output_path = "todo_data/03_straight_crawl/output_video.mp4"  # 생성될 영상 경로
# pcd_folder = "combined"
# output_folder = "combined/captured_images"
# output_video_path = "combined/output_video.mp4"

# 출력 폴더 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 공통 카메라 파라미터 설정 함수
def get_common_camera_parameters():
    lookat = np.array([0, 0, 0])  # 공통으로 설정할 바라보는 위치
    front = np.array([0, -1, 0])  # 기본적으로 뒤쪽에서 앞쪽을 바라보는 방향
    up = np.array([0, 0, 1])  # 위쪽 방향 (Z축이 위쪽)
    zoom = 0.1  # 줌 레벨 고정
    return lookat, front, up, zoom

# PCD 파일 불러오고 시각화하는 함수 (동적 카메라 파라미터 적용)
# def load_and_visualize_pcd(file_path, point_size=1.0):
# def load_and_visualize_pcd(file_path, point_size=1.0, capture_image=False, output_image_path=None):
# def load_and_capture_pcd_image(file_path, point_size=1.0, output_image_path=None):
def load_and_capture_pcd_image(pcd, bboxes, point_size=1.0, output_image_path=None, lookat=None, front=None, up=None, zoom=None):
    # pcd 파일 로드
    # pcd = o3d.io.read_point_cloud(file_path)
    # print(f"Point cloud has {len(pcd.points)} points.")

    # 동적으로 카메라 파라미터 설정
    bounding_box = pcd.get_axis_aligned_bounding_box()
    bbox_center = bounding_box.get_center()  # 바운딩 박스 중심
    # bbox_extent = bounding_box.get_extent()  # 바운딩 박스 크기

    # PCD를 z축 기준으로 시계방향으로 20도 회전
    # R = pcd.get_rotation_matrix_from_axis_angle(np.array([0, 0, np.deg2rad(-5)]))
    # pcd.rotate(R, center=bbox_center)


    # R_z = pcd.get_rotation_matrix_from_axis_angle(np.array([0, 0, np.deg2rad(20)]))
    # pcd.rotate(R_z, center=bbox_center)

    R_y = pcd.get_rotation_matrix_from_axis_angle(np.array([0, 0, np.deg2rad(10)]))
    pcd.rotate(R_y, center=bbox_center)

    # PCD를 zx 평면 기준으로 시계방향으로 5도 회전
    R_zx = pcd.get_rotation_matrix_from_axis_angle(np.array([np.deg2rad(5), 0, 0]))
    pcd.rotate(R_zx, center=bbox_center)
    # # 카메라 시점 조정: 바운딩 박스 중심을 기준으로 거리와 방향을 동적으로 설정
    # lookat = bbox_center
    # distance = np.linalg.norm(bbox_extent) * 2.0  # 바운딩 박스 크기에 비례한 거리 설정
    # front = np.array([0, -1, 0])  # 기본적으로 뒤쪽에서 앞쪽을 바라보는 방향
    # up = np.array([0, 0, 1])  # 위쪽 방향 (Z축이 위쪽)

    # 시각화 설정
    vis = o3d.visualization.Visualizer()
    # vis.create_window(visible=False)
    vis.create_window(visible=False, width=1920, height=1080) 
    # vis.create_window()
    vis.add_geometry(pcd)

    for bbox in bboxes:
                vis.add_geometry(bbox)

    # 카메라 파라미터 동적 적용
    view_control = vis.get_view_control()
    view_control.set_lookat(lookat)
    view_control.set_front(front)
    view_control.set_up(up)
    # view_control.set_zoom(0.35)  # 줌 레벨을 거리의 반비례로 설정하여 줌아웃 효과 적용
    view_control.set_zoom(zoom)

    vis.get_render_option().point_size = point_size
    vis.poll_events()
    vis.update_renderer()

    # 이미지 캡처 기능 추가
    # if capture_image and output_image_path:
    if output_image_path:
        vis.capture_screen_image(output_image_path)
        print(f"Captured image saved to {output_image_path}")

    # vis.run()
    vis.destroy_window()


# PCD 파일 불러오기 및 데이터 확인 함수
def load_and_inspect_pcd(file_path):
    # PCD 파일 로드
    pcd = o3d.io.read_point_cloud(file_path)
    
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

# 여러 이미지들을 영상으로 만드는 함수
def create_video_from_images(image_paths, video_output_path, fps=30):
    # 첫 번째 이미지로 프레임 크기 결정
    frame = cv2.imread(image_paths[0])
    height, width, _ = frame.shape
    
    # 비디오 작성기 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정 (MP4)
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    
    for idx, image_path in enumerate(image_paths, start=1):
        frame = cv2.imread(image_path)
        video_writer.write(frame)
        print(f"[{idx}/{len(image_paths)}] Adding frame: {image_path}")
    
    video_writer.release()
    print(f"Video saved to {video_output_path}")

def preprocess_and_generate_bounding_boxes(pcd):
    """
    PCD 데이터에 대해 전처리를 수행하고 바운딩 박스를 생성합니다.
    """
    # Voxel Downsampling
    voxel_size = 0.2
    downsample_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Radius Outlier Removal (ROR)
    cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    ror_pcd = downsample_pcd.select_by_index(ind)
    
    # RANSAC을 사용하여 평면 추정
    plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                                 ransac_n=3,
                                                 num_iterations=2000)
    final_point = ror_pcd.select_by_index(inliers, invert=True)
    
    # DBSCAN 클러스터링
    labels = np.array(final_point.cluster_dbscan(eps=0.35, min_points=10, print_progress=True))
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
    colors[labels < 0] = 0  # 노이즈는 검정색
    final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    # 클러스터 필터링 및 바운딩 박스 생성
    bboxes = []
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = final_point.select_by_index(cluster_indices)
        points = np.asarray(cluster_pcd.points)
        z_values = points[:, 2]
        
        if 5 <= len(cluster_indices) <= 40:  # 클러스터 크기 필터링
            z_min, z_max = z_values.min(), z_values.max()
            if -1.5 <= z_min <= 2.5 and 0.8 <= (z_max - z_min) <= 1.85:
                distances = np.linalg.norm(points, axis=1)
                if distances.max() <= 150.0:
                    bbox = cluster_pcd.get_axis_aligned_bounding_box()
                    bbox.color = (1, 0, 0)
                    bboxes.append(bbox)
    return final_point, bboxes


##  Model 불러오기



# 공통 카메라 파라미터 설정
lookat, front, up, zoom = get_common_camera_parameters()

image_paths = []
file_count = 0
for file_name in os.listdir(pcd_folder):
    # if file_name.endswith(".pcd"):
    if file_name.endswith(".pcd") :
        file_path = os.path.join(pcd_folder, file_name)
        output_image_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.png")
        # load_and_visualize_pcd(file_path, point_size=0.5, capture_image=True, output_image_path=output_image_path)
        # load_and_capture_pcd_image(file_path, point_size=0.5, output_image_path=output_image_path)
        pcd = o3d.io.read_point_cloud(file_path)
        prepro_pcd, bboxes = preprocess_and_generate_bounding_boxes(pcd)
        load_and_capture_pcd_image(prepro_pcd, bboxes, point_size=0.5, output_image_path=output_image_path, lookat=lookat, front=front, up=up, zoom=zoom)
        # load_and_inspect_pcd(file_path)
        image_paths.append(output_image_path)
        file_count += 1

# # pcd 시각화 및 이미지 캡처 테스트
# load_and_visualize_pcd(file_path, point_size=0.5, capture_image=True, output_image_path=output_image_path)
# # load_and_visualize_pcd(file_path, point_size=0.5)
# load_and_inspect_pcd(file_path)
# image_paths = [output_image_path]  # 여기서는 한 장의 이미지로 테스트, 여러 이미지가 있다면 리스트에 추가
create_video_from_images(image_paths, video_output_path, fps=20)