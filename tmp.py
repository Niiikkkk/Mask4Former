import numpy as np
import open3d as o3d

#path = "/normal_dataset/74/lidar/raw/lidar-21515.npy"
path = "/home/nicholas/mask4former/01/predictions/000000.label"
# label_path = path.replace("lidar", "semantic_lidar")
# #label_path = path.replace("velodyne", "labels").replace(".bin", ".label")
#
#x = np.load(path)
#points = np.frombuffer(x, np.float32).reshape(-1, 4)
#points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
#print(points)
#
# #lbl = np.load(label_path)
# # label_data_ = np.frombuffer(lbl, dtype=np.dtype([
# #     ('x', np.float32), ('y', np.float32), ('z', np.float32),
# #     ('intensity', np.float32), ('insta', np.uint32), ('sem', np.uint32)]))
# lbl = np.fromfile(label_path, dtype=np.int32)
#
# print(np.unique(lbl, return_counts=True))
#
#
# #label_data_ = np.array([label_data_['sem'][i] + (label_data_['insta'][i] << 16) for i in range(len(label_data_))])
#
# #tmp = np.unique(label_data_, return_counts=True)
#
# #print(tmp)
#
# #print(tmp[0][0] & 0xFFFF, tmp[1][0])
#
# filtered_data = [l for l in lbl if (l & 0xFFFF) == 10]
#
#
# print(np.unique(filtered_data, return_counts=True))
#
# # print(len(points[label_data_['sem'] == 12]))
#
pcd = o3d.geometry.PointCloud()

#print(np.unique(points[:,3], return_counts=True))

#pcd.points = o3d.utility.Vector3dVector(points[:, :3])
# o3d.visualization.draw_geometries_with_vertex_selection([pcd])

path = "/home/nicholas/Desktop/main_UE4/output/Train/1/lidar/raw/lidar-4436.npy"
#path = "/anomlay_carla/1/semantic_lidar/raw/semantic_lidar-3414.npy"
lidar_path = path.replace("lidar", "semantic_lidar")
# path = "/kitty/dataset/sequences/08/velodyne/000000.bin"
# path = "/normal_dataset/1/lidar/raw/lidar-13807.npy"
x = np.load(path)
y = np.load(lidar_path)
label_data_ = np.frombuffer(y, dtype=np.dtype([
    ('x', np.float32), ('y', np.float32), ('z', np.float32),
    ('intensity', np.float32), ('insta', np.uint32), ('sem', np.uint32)]))

label_data_ = np.array([label_data_['sem'][i] + (label_data_['insta'][i] << 16) for i in range(len(label_data_))])
#print(np.unique(label_data_))

points = np.frombuffer(x, dtype=np.dtype(np.float32)).reshape(-1, 4)
#print(label_data_.shape)
# points = np.fromfile(path, np.float32).reshape(-1, 4)
#points[:, 1] = -points[:, 1]  # Invert Y axis
print(points.shape)
print(label_data_.shape)
print(points[:,3].max(), points[:,3].min())

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.1)
print(voxel_grid)
is_occupied = voxel_grid.check_if_included(pcd.points)
print(np.unique(is_occupied,return_counts=True))

intensity = points[:, 3]

labels_colour = np.array([
    (0, 0, 0), # unlabeled = 0
    # cityscape
    (128, 64, 128), # road = 1
    (244, 35, 232), # sidewalk = 2
    (70, 70, 70), # bilding = 3
    (102, 102, 156), # wall = 4
    (190, 153, 153), # fence = 5
    (153, 153, 153), # pole = 6
    (250, 170, 30), # trafficlight = 7
    (220, 220, 0), # trafficsign = 8
    (107, 142, 35), # vegetation = 9
    (152, 251, 152), # terrain = 10
    (70, 130, 180), # sky = 11
    (220, 20, 60), # pedestrian = 12
    (255, 0, 0), # rider = 13
    (0, 0, 142), # Car = 14
    (0, 0, 70), # trck = 15
    (0, 60, 100), # bs = 16
    (0, 80, 100), # train = 17
    (0, 0, 230), # motorcycle = 18
    (119, 11, 32), # bicycle = 19
                                   # cstom
    (110, 190, 160), # static = 20
    (170, 120, 50), # dynamic = 21
    (55, 90, 80), # other = 22
    (45, 60, 150), # water = 23
    (157, 234, 50), # roadline = 24
    (81, 0, 81), # grond = 25
    (150, 100, 100), # bridge = 26
    (230, 150, 140), # railtrack = 27
    (180, 165, 180), # gardrail = 28
    (180, 130, 70), # rock = 29
        #anomalies
    (193, 71, 71), # Static_Anomaly = 30
    (102, 102, 255), # Dynamic_Anomaly = 31
    (175, 83, 83), # Animal = 32
    (232, 188, 188), # Tinyanomaly = 33
    (229, 137, 137), # Smallanomaly = 34
    (189, 47, 47), # Mediumanomaly = 35
    (131, 7, 7), # Largeanomaly = 36
    ]) / 255.0

colors = labels_colour[label_data_ & 0xFFFF]

pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries_with_vertex_selection([pcd])


pred_label_path = "/home/nicholas/mask4former/01/predictions/000000.label"
file = np.fromfile(pred_label_path, dtype=np.uint32)
lbls = file & 0xFFFF

colors = labels_colour[lbls]
pcd.colors = o3d.utility.Vector3dVector(colors)
#o3d.visualization.draw_geometries_with_vertex_selection([pcd])

