import sys
import os

import torch
import trimesh
import numpy as np
import pyviz3d.visualizer as viz
from hy3dshape.hy3dshape.models.autoencoders import ShapeVAE

def run_reconstruction():
    """
    一个完整的示例，展示如何使用ShapeVAE直接从点云重建3D模型。
    """
    # --- 1. 配置 ---
    # input_mesh_path = "assets/1.glb"
    # input_mesh_path = "assets/part1.stl"
    input_mesh_path = "/home/ljr/Hunyuan3D-2.1/data/abc/abc_obj/00000020_ad34a3f60c4a4caa99646600_trimesh_000.obj"
    # input_mesh_path = "assets/abc.obj"
    output_mesh_path = "reconstructed_mesh.glb"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    # VAE编码器期望的点云大小，这里参考了encoder中的默认值
    total_size = 81920
    pc_size = 81920  # 表面随机采样点数
    pc_sharpedge_size = total_size-pc_size  # 边缘采样点数
    num_points_for_encoder = pc_size + pc_sharpedge_size
    edge_angle_threshold_deg = 45.0  # 用于定义锐利边缘的角度阈值（度）

    print("--- 开始3D重建任务 ---")
    print(f"设备: {device}, 数据类型: {dtype}")

    # --- 2. 加载独立的ShapeVAE模型 ---
    print(f"正在加载 ShapeVAE 模型...")
    try:
        vae = ShapeVAE.from_pretrained(
            'tencent/Hunyuan3D-2.1',
            subfolder='hunyuan3d-vae-v2-1',
            device=device,
            dtype=dtype
        )
        vae.eval() # 设置为评估模式
        print("ShapeVAE 模型加载成功。")
    except Exception as e:
        print(f"加载模型失败，请确保网络连接正常或模型文件已缓存。错误: {e}")
        return

    # --- 3. 加载并准备输入点云 ---
    print(f"正在从 '{input_mesh_path}' 加载输入模型...")
    try:
        # 使用 trimesh 加载任意格式的3D文件
        input_mesh = trimesh.load(input_mesh_path, force='mesh')
        # 确保法线存在
        input_mesh.vertex_normals
    except Exception as e:
        print(f"加载输入模型失败。错误: {e}")
        return

    # --- 3.1 (新增) 计算并打印坐标范围 ---
    # print("计算模型坐标范围...")
    min_bounds, max_bounds = input_mesh.bounds
    # print(f"模型最小坐标: {min_bounds}")
    # print(f"模型最大坐标: {max_bounds}")

    # 根据模型的边界来归一化点云
    # 1. 计算中心点和尺寸
    center = (min_bounds + max_bounds) / 2.0
    scale = (max_bounds - min_bounds).max()

    # 2. 归一化顶点
    normalized_vertices = (input_mesh.vertices - center) / scale
    input_mesh.vertices = normalized_vertices

    print(f"正在从模型表面采样 {pc_size} 个随机点...")
    points_surface, face_indices_surface = input_mesh.sample(pc_size, return_index=True)
    normals_surface = input_mesh.face_normals[face_indices_surface]
    labels_surface = np.zeros((pc_size, 1), dtype=np.float32)

    # --- 3.2 在锐利边缘上采样点 ---
    print(f"正在从锐利边缘采样 {pc_sharpedge_size} 个点 (角度阈值: {edge_angle_threshold_deg}°)...")
    # 查找所有角度大于阈值的边
    edge_angles = input_mesh.face_adjacency_angles
    sharp_edge_indices = np.where(np.rad2deg(edge_angles) > edge_angle_threshold_deg)[0]
    
    if len(sharp_edge_indices) > 0:
        edges_sharp = input_mesh.face_adjacency_edges[sharp_edge_indices]
        
        # 计算每条锐利边的长度
        edge_vertices = input_mesh.vertices[edges_sharp]
        edge_lengths = np.linalg.norm(edge_vertices[:, 0] - edge_vertices[:, 1], axis=1)
        
        # 根据边长进行加权采样
        probabilities = edge_lengths / edge_lengths.sum()
        sampled_edge_indices = np.random.choice(
            np.arange(len(edges_sharp)),
            size=pc_sharpedge_size,
            p=probabilities
        )
        
        # 在采样到的边上生成点
        t = np.random.rand(pc_sharpedge_size, 1)
        points_edge = (1 - t) * edge_vertices[sampled_edge_indices, 0] + t * edge_vertices[sampled_edge_indices, 1]
        
        # 对于边缘点，法线可以从相邻面插值得到，这里简化处理，使用临近表面点的法线
        # 找到每个边缘点最近的表面点来获取法线
        from scipy.spatial import cKDTree
        tree = cKDTree(points_surface)
        _, closest_indices = tree.query(points_edge)
        normals_edge = normals_surface[closest_indices]
        
        labels_edge = np.ones((pc_sharpedge_size, 1), dtype=np.float32)
        
        # 合并表面点和边缘点
        points = np.concatenate([points_surface, points_edge], axis=0)
        normals = np.concatenate([normals_surface, normals_edge], axis=0)
        sharp_edge_labels = np.concatenate([labels_surface, labels_edge], axis=0)
    else:
        print("警告: 未找到锐利边缘。将使用100%的表面采样点。")
        # 如果没有锐利边缘，则全部使用表面采样
        points, face_indices = input_mesh.sample(num_points_for_encoder, return_index=True)
        normals = input_mesh.face_normals[face_indices]
        sharp_edge_labels = np.zeros((num_points_for_encoder, 1), dtype=np.float32)


    # --- 3.5 (可选) 给输入点云添加噪声 ---
    # noise_std: 高斯噪声的标准差
    # dropout_ratio: 随机丢弃点的比例
    noise_std = 0.01
    dropout_ratio = 0.2
    
    print(f"正在为点云添加噪声: std={noise_std}, dropout={dropout_ratio}")
    
    # 1. 添加高斯噪声到坐标
    noise = np.random.normal(0, noise_std, points.shape)
    noisy_points = points + noise
    
    # 将坐标、法线和锐利边缘标签合并成 (N, 7) 的形状
    # 尝试反转输入的法线方向
    point_cloud_data = np.concatenate([noisy_points, np.zeros_like(normals), sharp_edge_labels], axis=1)

    # 转换为PyTorch张量，并增加batch维度
    point_cloud_tensor = torch.from_numpy(point_cloud_data).unsqueeze(0).to(device=device, dtype=dtype)
    # --- 3.6 (新增) 使用 Pyviz3d 可视化 ---
    print("正在使用 Pyviz3d 生成可视化文件...")
    v = viz.Visualizer()
    
    # 为不同标签的点分配颜色
    # 锐利边缘 (label=1) 为红色, 表面 (label=0) 为蓝色
    point_colors = np.where(sharp_edge_labels > 0.5, [255, 0, 0], [0, 0, 255])
    
    # 可视化带噪声的点
    v.add_points("Noisy Point Cloud", noisy_points, colors=point_colors, point_size=1)
    
    # 可视化法线 (为了清晰，可以只显示一部分)
    sample_indices = np.random.choice(noisy_points.shape[0], 5000, replace=False)
    start_points = noisy_points[sample_indices]
    end_points = start_points + normals[sample_indices] * 0.02  # 调整法线长度
    v.add_lines("Normals", start_points, end_points)
    
    # 保存可视化结果
    viz_path = "point_cloud_visualization"
    v.save(viz_path)
    print(f"可视化结果已保存到: {viz_path}")

    print(f"点云数据准备完成，张量形状: {point_cloud_tensor.shape}")

    print(vae.volume_decoder)

    # --- 4. 执行编码和解码 ---
    with torch.no_grad():
        # 编码：将点云压缩为潜在向量
        print("正在编码点云到潜在空间...")
        latents = vae.encode(point_cloud_tensor, sample_posterior=True)
        print(f"编码完成，潜在向量形状: {latents.shape}")

        # 解码：将潜在向量重建为3D网格
        print("正在从潜在向量解码为3D网格...")
        # 1. 将64维的latent扩展到解码器期望的1024维
        decoded_latents =   vae.decode(latents)
        print(f"Latent解码完成，扩展后形状: {decoded_latents.shape}")

        # 2. 使用扩展后的latent生成网格
        # latents2mesh 是解码流程的统一入口
        mesh_list = vae.latents2mesh(
            decoded_latents,
            octree_resolution=512, # 可以调整分辨率
            num_chunks=10000,
            enable_pbar=True,
            mc_level=0.0,
            bounds=[-1.01, -1.01, -1.01, 1.01, 1.01, 1.01]
        )
        print("解码完成。")

    # --- 5. 保存输出结果 ---
    if mesh_list and mesh_list[0] is not None:
        reconstructed_mesh_data = mesh_list[0]
        
        # 使用trimesh创建一个新的mesh对象并导出
        final_mesh = trimesh.Trimesh(
            vertices=reconstructed_mesh_data.mesh_v,
            faces=reconstructed_mesh_data.mesh_f
        )
        
        # 修复可能反向的法线
        # final_mesh.fix_normals()
        final_mesh.export(output_mesh_path)
        print(f"重建的3D模型已成功保存到: {output_mesh_path}")
    else:
        print("解码失败，未能生成有效的网格。")

if __name__ == "__main__":
    run_reconstruction()
