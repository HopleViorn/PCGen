import numpy as np
import trimesh
import pyviz3d.visualizer as vis
from scipy.spatial import cKDTree

def get_l1_median(points, num_iterations=100, tolerance=1e-5):
    """
    计算点集的L1中值（几何中值）。
    :param points: 一个 (n, 3) 的 numpy 数组，包含n个点的坐标。
    :param num_iterations: 最大迭代次数。
    :param tolerance: 收敛容差。
    :return: L1中值的坐标 (x, y, z)。
    """
    # 初始化中值为所有点的几何中心
    median = np.mean(points, axis=0)
    
    for _ in range(num_iterations):
        prev_median = median.copy()
        
        # 计算每个点到当前中值的距离
        distances = np.linalg.norm(points - median, axis=1)
        
        # 避免除以零（如果中值与某个点重合）
        distances[distances < 1e-9] = 1e-9
        
        # 计算权重（距离的倒数）
        weights = 1.0 / distances
        
        # 更新中值（加权平均）
        median = np.sum(points * weights[:, np.newaxis], axis=0) / np.sum(weights)
        
        # 检查收敛性
        if np.linalg.norm(median - prev_median) < tolerance:
            break
            
    return median

def compute_l1_skeleton(points, num_samples=1000, k_neighbors=10, num_iterations=100, tolerance=1e-5):
    """
    通过在点云上采样局部邻域并计算L1中值来计算L1-Skeleton。
    :param points: 输入点云 (N, 3)。
    :param num_samples: 要采样的局部邻域数量。
    :param k_neighbors: 每个局部邻域中的点的数量。
    :param num_iterations: L1中值计算的迭代次数。
    :param tolerance: L1中值计算的收敛容差。
    :return: L1-Skeleton点 (num_samples, 3)。
    """
    print(f"正在构建KDTree以便进行高效的邻域搜索...")
    kdtree = cKDTree(points)
    
    # 在点云中随机选择采样点
    sample_indices = np.random.choice(len(points), num_samples, replace=False)
    sampled_points = points[sample_indices]
    
    skeleton_points = []
    print(f"开始为 {num_samples} 个采样点计算L1中值...")
    
    for i, point in enumerate(sampled_points):
        # 查找k个最近邻
        distances, indices = kdtree.query(point, k=k_neighbors)
        
        # 提取邻域点
        neighborhood = points[indices]
        
        # 计算该邻域的L1中值
        l1_median = get_l1_median(neighborhood, num_iterations, tolerance)
        skeleton_points.append(l1_median)
        
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{num_samples} 个点...")
            
    return np.array(skeleton_points)

def main(ply_file_path):
    """
    主函数，用于加载点云，计算L1-Skeleton并进行可视化。
    :param ply_file_path: 输入的 .ply 文件路径。
    """
    # 1. 加载点云
    try:
        point_cloud = trimesh.load(ply_file_path)
        points = point_cloud.vertices
        print(f"成功加载点云，包含 {len(points)} 个点。")
    except Exception as e:
        print(f"加载文件 '{ply_file_path}' 失败: {e}")
        return

    # 1.5 对点云进行归一化处理
    shifts = (points.max(axis=0) + points.min(axis=0)) / 2
    points = points - shifts
    distances = np.linalg.norm(points, axis=1)
    scale = 1 / np.max(distances)
    points *= scale
    print("点云已归一化处理")

    # 2. 计算L1-Skeleton
    print("正在计算L1-Skeleton...")
    # 增大邻域范围（k_neighbors），以确保骨架点在物体内部生成
    k = len(points) // 4
    print(f"使用 k = {k} 作为邻域点数")
    skeleton_points = compute_l1_skeleton(points, num_samples=2000, k_neighbors=k)
    print(f"成功计算了 {len(skeleton_points)} 个L1-Skeleton点。")

    # 保存L1-Skeleton为 .ply 文件
    if len(skeleton_points) > 0:
        skeleton_cloud = trimesh.PointCloud(skeleton_points)
        skeleton_cloud.export('l1_skeleton.ply')
        print(f"L1-Skeleton已保存至 'l1_skeleton.ply'")
    else:
        print("没有有效的L1-Skeleton点可保存。")

    # 3. 使用 pyviz3d 进行可视化
    print("正在生成可视化文件...")
    v = vis.Visualizer()
    
    # 可视化原始点云
    v.add_points("Original Points", points, colors=np.array([[255, 0, 0]]).repeat(len(points),axis=0), point_size=5)
    
    # 可视化L1-Skeleton
    if len(skeleton_points) > 0:
        v.add_points("L1 Skeleton", skeleton_points, colors=np.array([[0, 0, 255]]).repeat(len(skeleton_points),axis=0), point_size=10)

    # 保存为 HTML 文件
    output_filename = "l1_skeleton_visualization"
    v.save(output_filename)
    print(f"可视化结果已保存至 '{output_filename}'。请在浏览器中打开查看。")


if __name__ == '__main__':
    # 请将 'input.ply' 替换为您的点云文件路径
    input_file = 'test.ply'
    print(f"输入文件: {input_file}")
    
    main(input_file)