import numpy as np
import trimesh
from scipy.spatial import Delaunay
import pyviz3d.visualizer as vis

def get_circumcenter(tetrahedron_points):
    """
    计算四面体的外接球球心。
    :param tetrahedron_points: 一个 (4, 3) 的 numpy 数组，包含四面体四个顶点的坐标。
    :return: 外接球球心的坐标 (x, y, z)。
    """
    # 构造矩阵 A
    A = np.zeros((4, 4))
    for i in range(4):
        A[i, 0] = 2 * tetrahedron_points[i, 0]
        A[i, 1] = 2 * tetrahedron_points[i, 1]
        A[i, 2] = 2 * tetrahedron_points[i, 2]
        A[i, 3] = 1

    # 构造向量 b
    b = np.array([np.sum(p**2) for p in tetrahedron_points])

    # 解线性方程组 Ax = b，其中 x = [cx, cy, cz, r^2 - cx^2 - cy^2 - cz^2]
    # 我们只需要前三个分量
    try:
        x = np.linalg.solve(A, b)
        return x[:3]
    except np.linalg.LinAlgError:
        # 如果矩阵是奇异的（例如，四个点共面），则返回质心作为近似
        return np.mean(tetrahedron_points, axis=0)

def main(ply_file_path):
    """
    主函数，用于加载点云，计算德劳内三角剖分，找到外接球球心并进行可视化。
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

    # 2. 进行 Delaunay 三角剖分
    print("正在进行 Delaunay 三角剖分...")
    delaunay = Delaunay(points)
    tetrahedrons = delaunay.simplices
    print(f"生成了 {len(tetrahedrons)} 个四面体。")

    # 3. 计算所有四面体的外接球球心
    print("正在计算外接球球心...")
    circumcenters = np.array([get_circumcenter(points[simplex]) for simplex in tetrahedrons])
    
    # 过滤掉计算失败或无穷大的球心
    valid_centers = circumcenters[np.all(np.isfinite(circumcenters), axis=1)]
    print(f"成功计算了 {len(valid_centers)} 个有效的外接球球心。")
    
    # 过滤掉距离原点超过一定距离的点
    distance_threshold = 1.0  # 距离阈值
    distances = np.linalg.norm(valid_centers, axis=1)  # 计算每个点到原点的距离
    filtered_centers = valid_centers[distances <= distance_threshold]  # 过滤掉超过阈值的点
    
    print(f"过滤掉了 {len(valid_centers) - len(filtered_centers)} 个距离原点超过 {distance_threshold} 的点")
    print(f"剩余 {len(filtered_centers)} 个点用于保存")

    # 保存外接球球心为 .ply 文件
    if len(filtered_centers) > 0:
        point_cloud = trimesh.PointCloud(filtered_centers)
        point_cloud.export('circumcenters.ply')
        print(f"外接球球心已保存至 'circumcenters.ply'")
    else:
        print("没有有效的外接球球心可保存。")

    # 4. 使用 pyviz3d 进行可视化
    print("正在生成可视化文件...")
    v = vis.Visualizer()
    
    # 可视化原始点云
    v.add_points("Original Points", points, colors=np.array([[255, 0, 0]]).repeat(len(points),axis=0), point_size=5)
    

    # 可视化外接球球心
    if len(valid_centers) > 0:
        v.add_points("Circumcenters", valid_centers, colors=np.array([[0, 0, 255]]).repeat(len(valid_centers),axis=0), point_size=5)

    # # 可视化 Delaunay 三角剖分的线框
    # # 提取四面体的所有边
    # edges = set()
    # for simplex in tetrahedrons:
    #     # 四面体的6条边: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    #     edges.add((simplex[0], simplex[1]))
    #     edges.add((simplex[0], simplex[2]))
    #     edges.add((simplex[0], simplex[3]))
    #     edges.add((simplex[1], simplex[2]))
    #     edges.add((simplex[1], simplex[3]))
    #     edges.add((simplex[2], simplex[3]))
    
    # # 将边转换为线段的起点和终点
    # lines_start = np.array([points[edge[0]] for edge in edges])
    # lines_end = np.array([points[edge[1]] for edge in edges])
    
    # # 添加线框可视化
    # v.add_lines("Delaunay Triangulation", lines_start, lines_end, colors=np.array([[0, 255, 0]]).repeat(len(edges), axis=0))

    # 保存为 HTML 文件
    output_filename = "delaunay_visualization"
    v.save(output_filename)
    print(f"可视化结果已保存至 '{output_filename}'。请在浏览器中打开查看。")


if __name__ == '__main__':
    # 请将 'input.ply' 替换为您的点云文件路径
    # input_file = '/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/data/pc_cad/0003/00030536.ply' 
    # input_file = '/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/data/pc_cad/0003/00039972.ply' 
    # input_file = '/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/data/pc_cad/0003/00038036.ply'
    # input_file = '/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/data/pc_cad/0078/00789510.ply'
    # input_file = '/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/data/pc_cad/0074/00740902.ply'
    # input_file = '/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/data/pc_cad/0083/00830418.ply'
    input_file = 'test.ply'
    # input_file = '/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/data/pc_cad/0043/00430264.ply'
    print(input_file)
    
    main(input_file)