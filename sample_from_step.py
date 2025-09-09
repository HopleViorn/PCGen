import argparse
import trimesh
import numpy as np
from plyfile import PlyData, PlyElement
import os
# For parametric sampling, python-occ-core is required.
# You can install it via conda: conda install -c conda-forge python-occ-core
try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_REVERSED
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.gp import gp_Pnt, gp_Pnt2d, gp_Vec
    from OCC.Core.BRepClass import BRepClass_FaceClassifier
    import random
    PYTHON_OCC_AVAILABLE = True
except ImportError:
    PYTHON_OCC_AVAILABLE = False

def write_ply(points, filename, normals=None, text=False):
    """
    将点和法线（可选）写入PLY文件。
    """
    if normals is not None and len(points) == len(normals):
        # Create a structured array with points and normals
        vertex_data = np.hstack([points, normals])
        vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
    else:
        # Create a structured array with only points
        vertex_data = points
        vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    
    vertex = np.array([tuple(row) for row in vertex_data], dtype=vertex_dtype)
    el = PlyElement.describe(vertex, 'vertex')
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)

def main():
    """
    主函数，用于加载STEP文件，采样点并保存为PLY文件。
    """
    parser = argparse.ArgumentParser(description='Sample points on the surface of a STEP file.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input .step file.')
    parser.add_argument('--num_points', type=int, required=True, help='Number of points to sample.')
    parser.add_argument('--output_file', type=str, help='Path to the output .ply file. If not provided, it will be generated from the input file name.')
    parser.add_argument('--method', type=str, default='parametric', choices=['mesh', 'parametric'], help='Sampling method: "mesh" (faster, approximate) or "parametric" (slower, more accurate).')
    parser.add_argument('--with_normals', action='store_true', help='If set, sample surface normals along with points.')
    
    args = parser.parse_args()

    input_path = args.input_file
    num_points = args.num_points
    output_path = args.output_file
    method = args.method

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    if method == 'parametric' and not PYTHON_OCC_AVAILABLE:
        print("Error: 'parametric' method requires python-occ-core, but it's not installed.")
        print("Please install it via: conda install -c conda-forge python-occ-core")
        return

    # 如果未指定输出文件，则根据输入文件生成
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        normals_suffix = "_with_normals" if args.with_normals else ""
        output_path = f"{base_name}_{num_points}_{method}{normals_suffix}.ply"

    points, normals = None, None
    if method == 'mesh':
        print(f"Using 'mesh' sampling method (via trimesh)...")
        points, normals = sample_mesh_surface(input_path, num_points, args.with_normals)
    elif method == 'parametric':
        print(f"Using 'parametric' sampling method (via python-occ)...")
        points, normals = sample_parametric_surface(input_path, num_points, args.with_normals)

    if points is not None and len(points) > 0:
        print(f"Saving {len(points)} sampled points to {output_path}...")
        # 保存为PLY文件
        write_ply(points, output_path, normals=normals)
        print("Done.")
    else:
        print("No points were sampled. Exiting.")

def sample_mesh_surface(file_path, num_points, with_normals=False):
    """
    使用trimesh从网格表面采样点。
    """
    print(f"Loading mesh from {file_path}...")
    try:
        # force='mesh' is important to ensure STEP files are converted to a mesh
        mesh = trimesh.load(file_path, force='mesh')
    except Exception as e:
        print(f"Failed to load mesh with trimesh: {e}")
        return None, None
        
    print(f"Sampling {num_points} points from the mesh surface...")
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    
    normals = None
    if with_normals:
        print("Calculating normals for sampled points...")
        normals = mesh.face_normals[face_indices]

    return points, normals

def sample_parametric_surface(file_path, num_points, with_normals=False):
    """
    使用python-occ直接从参数化曲面采样点。
    """
    print(f"Loading shape from {file_path}...")
    reader = STEPControl_Reader()
    status = reader.ReadFile(file_path)
    if status != 1: # IFSelect_RetDone
        print("Error: Could not read STEP file.")
        return None, None
    reader.TransferRoots()
    shape = reader.Shape()

    # 遍历所有面并计算总面积
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []
    areas = []
    props = GProp_GProps()
    while face_explorer.More():
        face = face_explorer.Current()
        brepgprop_SurfaceProperties(face, props)
        area = props.Mass()
        if area > 1e-6: # 忽略面积非常小的面
            faces.append(face)
            areas.append(area)
        face_explorer.Next()

    if not faces:
        print("No faces with positive area found in the shape.")
        return None, None

    total_area = sum(areas)
    areas = np.array(areas)
    points = []
    normals = [] if with_normals else None

    print(f"Sampling from {len(faces)} faces with total area {total_area:.2f}...")
    for i, face in enumerate(faces):
        num_face_points = int(np.round(num_points * (areas[i] / total_area)))
        if num_face_points == 0:
            continue

        surf = BRepAdaptor_Surface(face, True)
        umin, umax, vmin, vmax = surf.FirstUParameter(), surf.LastUParameter(), surf.FirstVParameter(), surf.LastVParameter()
        
        classifier = BRepClass_FaceClassifier()
        
        count = 0
        max_tries = num_face_points * 100 # 安全退出机制
        tries = 0
        while count < num_face_points and tries < max_tries:
            u = random.uniform(umin, umax)
            v = random.uniform(vmin, vmax)
            
            # 检查点是否在面的边界内
            classifier.Perform(face, gp_Pnt2d(u, v), 1e-6)
            
            if classifier.State() == TopAbs_IN:
                pnt = gp_Pnt()
                if with_normals:
                    D1u = gp_Vec()
                    D1v = gp_Vec()
                    surf.D1(u, v, pnt, D1u, D1v)
                    
                    normal_vec = D1u.Crossed(D1v)
                    if normal_vec.Magnitude() > 1e-9: # Avoid zero-length normals
                        normal_vec.Normalize()
                    
                    if face.Orientation() == TopAbs_REVERSED:
                        normal_vec.Reverse()

                    normals.append([normal_vec.X(), normal_vec.Y(), normal_vec.Z()])
                else:
                    surf.D0(u, v, pnt)

                points.append([pnt.X(), pnt.Y(), pnt.Z()])
                count += 1
            tries += 1
        
        if tries >= max_tries and num_face_points > 0:
            print(f"Warning: Could only sample {count}/{num_face_points} points for a face after {max_tries} attempts.")

    points_arr = np.array(points)
    normals_arr = np.array(normals) if with_normals else None

    print(normals_arr)
    return points_arr, normals_arr

if __name__ == '__main__':
    main()