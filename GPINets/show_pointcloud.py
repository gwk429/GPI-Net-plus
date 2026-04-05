
import random


import json
import copy

from matplotlib import pyplot as plt

from utils.pointcloud import estimate_normal

import numpy as np
import open3d as o3d


def visual(source, target, corr):
    # 读取NPZ文件
    data = np.load(source)
    t = list(data.keys())
    points = data['pcd']
    colors = data['color']  # 如果有颜色信息的话

    # 创建PointCloud对象
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points)
    pcd1.colors = o3d.utility.Vector3dVector(colors)

    data = np.load(target)
    points = data['pcd']
    colors = data['color']  # 如果有颜色信息的话

    # 创建PointCloud对象
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points)
    pcd2.colors = o3d.utility.Vector3dVector(colors)

    # 将第二个点云沿 x 轴方向平移一定距离
    translation_distance = 4  # 平移距离
    translation_vector = np.array([translation_distance, 0, 0])  # 平移向量
    pcd2.translate(translation_vector)  # 对 pcd2 进行平移

    corr = corr.tolist()

    num_to_delete = int(len(corr) * 0.7)

    # 随机选择要删除的元素的索引
    indexes_to_delete = random.sample(range(len(corr)), num_to_delete)

    # 删除选定的元素
    for index in sorted(indexes_to_delete, reverse=True):
        del corr[index]

    # ---------------------------------------------获取直线-----------------------------------------------
    line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pcd1, pcd2, corr)
    # 给对应线赋色，方法一
    line_set.paint_uniform_color([1, 0, 0])
    # ---------------------------------------------结果可视化---------------------------------------------
    o3d.visualization.draw_geometries([pcd1, pcd2, line_set])


def draw_registration_result(source, target, transformation, i):
    source = o3d.io.read_point_cloud(source)
    target = o3d.io.read_point_cloud(target)

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if not source_temp.has_normals():
        estimate_normal(source_temp)
        estimate_normal(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

    # 可视化并保存为 PNG
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    # 进入交互式模式
    vis.run()  # 运行交互式可视化，手动调整视角
    view_ctl = vis.get_view_control()
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    # 保存 PNG 文件
    path = f"E:/download_browser/PG-Net-master/vis/{i}_.png"
    plt.imsave(path, np.asarray(image))


def draw_registration_result1(source, target, transformation, corr, corr1):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if not source_temp.has_normals():
        estimate_normal(source_temp)
        estimate_normal(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    translation_distance = 4  # 平移距离
    translation_vector = np.array([translation_distance, 0, 0])  # 平移向量
    target_temp.translate(translation_vector)  # 对 pcd2 进行平移

    corr = corr.tolist()
    num_to_delete = int(len(corr) * 0.99)
    # 随机选择要删除的元素的索引
    indexes_to_delete = random.sample(range(len(corr)), num_to_delete)
    # 删除选定的元素
    for index in sorted(indexes_to_delete, reverse=True):
        del corr[index]

    line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(source_temp, target_temp, corr)
    line_set.paint_uniform_color([1, 0, 0])

    corr1 = corr1.tolist()
    num_to_delete = int(len(corr1) * 0.99)
    # 随机选择要删除的元素的索引
    indexes_to_delete = random.sample(range(len(corr1)), num_to_delete)
    # 删除选定的元素
    for index in sorted(indexes_to_delete, reverse=True):
        del corr1[index]

    line_set1 = o3d.geometry.LineSet.create_from_point_cloud_correspondences(source_temp, target_temp, corr1)
    line_set1.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([source_temp, target_temp, line_set])
    o3d.visualization.draw_geometries([source_temp, target_temp, line_set1])


def draw_registration_result2(source, target, transformation, corr, corr1, i, wrong_corr):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if not source_temp.has_normals():
        estimate_normal(source_temp)
        estimate_normal(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    translation_distance = 3  # 平移距离
    translation_vector = np.array([0, translation_distance, 0])  # 平移向量
    target_temp.translate(translation_vector)  # 对 pcd2 进行平移

    corr = corr.tolist()
    num_to_delete = int(len(corr) * 0.95)
    # 随机选择要删除的元素的索引
    indexes_to_delete = random.sample(range(len(corr)), num_to_delete)
    # 删除选定的元素
    for index in sorted(indexes_to_delete, reverse=True):
        del corr[index]
    line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(source_temp, target_temp, corr)
    line_set.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([source_temp, target_temp, line_set])

    corr1 = corr1.tolist()
    num_to_delete = int(len(corr1) * 0.3)
    # 随机选择要删除的元素的索引
    indexes_to_delete = random.sample(range(len(corr1)), num_to_delete)
    # 删除选定的元素
    for index in sorted(indexes_to_delete, reverse=True):
        del corr1[index]
    line_set1 = o3d.geometry.LineSet.create_from_point_cloud_correspondences(source_temp, target_temp, corr1)
    line_set1.paint_uniform_color([0, 1, 0])

    w_corr = wrong_corr.tolist()
    num_to_delete = int(len(w_corr) * 0.999)
    # 随机选择要删除的元素的索引
    indexes_to_delete = random.sample(range(len(w_corr)), num_to_delete)
    # 删除选定的元素
    for index in sorted(indexes_to_delete, reverse=True):
        del w_corr[index]
    line_set2 = o3d.geometry.LineSet.create_from_point_cloud_correspondences(source_temp, target_temp, w_corr)
    line_set2.paint_uniform_color([1, 0, 0])

    # 将 points 和 colors 从 LineSet 对象中提取为 NumPy 数组
    points1 = np.asarray(line_set1.points)
    points2 = np.asarray(line_set2.points)
    colors1 = np.asarray(line_set1.colors)
    colors2 = np.asarray(line_set2.colors)
    line1 = np.asarray(line_set1.lines)
    line2 = np.asarray(line_set2.lines)

    # 合并 points 和 colors
    combined_points = np.vstack((points1, points2))
    combined_colors = np.vstack((colors1, colors2))
    combined_lines = np.vstack((line1, line2))

    combined_line_set = o3d.geometry.LineSet()
    combined_line_set.points = o3d.utility.Vector3dVector(combined_points)
    combined_line_set.colors = o3d.utility.Vector3dVector(combined_colors)
    combined_line_set.lines = o3d.utility.Vector2iVector(combined_lines)

    o3d.visualization.draw_geometries([source_temp, target_temp, combined_line_set])

    # 可视化并保存为 PNG
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    vis.add_geometry(combined_line_set)
    # 进入交互式模式
    vis.run()  # 运行交互式可视化，手动调整视角
    view_ctl = vis.get_view_control()
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    # 保存 PNG 文件
    path = f"F:/gwk/EMSBC/PointDSC-master/vis/{i}.png"
    plt.imsave(path, np.asarray(image))


def computer_corr(frag1, frag2, srcc, tgtt):
    indices1 = np.zeros(len(frag1), dtype=int)
    indices2 = np.zeros(len(frag2), dtype=int)

    for i in range(len(frag1)):
        # 计算 a[i] 与 b 中每行的欧氏距离
        distances = np.linalg.norm(srcc - frag1[i], axis=1)
        # 找到距离最小的位置索引
        index = np.argmin(distances)
        value = np.min(distances)
        # 存储索引值
        indices1[i] = index

    for i in range(len(frag2)):
        # 计算 a[i] 与 b 中每行的欧氏距离
        distances = np.linalg.norm(tgtt - frag2[i], axis=1)
        # 找到距离最小的位置索引
        index = np.argmin(distances)
        # 存储索引值
        indices2[i] = index

    merged_array = [[0] * 2 for _ in range(len(frag2))]
    # 将第一个数组的元素复制到新数组的第一列
    for i in range(len(frag2)):
        merged_array[i][0] = indices1[i]

    # 将第二个数组的元素复制到新数组的第二列
    for i in range(len(frag2)):
        merged_array[i][1] = indices2[i]

    corr = np.array(merged_array)

    return corr
