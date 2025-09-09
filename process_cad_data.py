import argparse
import trimesh
import numpy as np
from plyfile import PlyData, PlyElement
import os
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

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
    os.makedirs(os.path.dirname(filename), exist_ok=True)
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

def sample_mesh_surface(file_path, num_points, with_normals=False):
    """
    使用trimesh从网格表面采样点。
    """
    try:
        # force='mesh' is important to ensure STEP files are converted to a mesh
        mesh = trimesh.load(file_path, force='mesh')
    except Exception as e:
        print(f"Failed to load mesh with trimesh: {e}")
        return None, None
        
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    
    normals = None
    if with_normals:
        normals = mesh.face_normals[face_indices]

    return points, normals

def sample_parametric_surface(file_path, num_points, with_normals=False):
    """
    使用python-occ直接从参数化曲面采样点。
    """
    reader = STEPControl_Reader()
    status = reader.ReadFile(file_path)
    if status != 1: # IFSelect_RetDone
        print(f"Error: Could not read STEP file: {file_path}")
        return None, None
    reader.TransferRoots()
    try:
        shape = reader.Shape()
    except RuntimeError as e:
        print(f"Error processing shape for {file_path}: {e}")
        return None, None

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
        print(f"No faces with positive area found in the shape: {file_path}")
        return None, None

    total_area = sum(areas)
    areas = np.array(areas)
    points = []
    normals = [] if with_normals else None

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
            print(f"Warning: Could only sample {count}/{num_face_points} points for a face in {file_path} after {max_tries} attempts.")

    points_arr = np.array(points)
    normals_arr = np.array(normals) if with_normals else None

    return points_arr, normals_arr

def process_file(input_path, output_path, num_points, method, with_normals):
    """
    处理单个STEP文件并保存为PLY。
    """
    if os.path.exists(output_path):
        # print(f"Output file already exists, skipping: {output_path}")
        return

    points, normals = None, None
    if method == 'mesh':
        points, normals = sample_mesh_surface(input_path, num_points, with_normals)
    elif method == 'parametric':
        points, normals = sample_parametric_surface(input_path, num_points, with_normals)

    if points is not None and len(points) > 0:
        write_ply(points, output_path, normals=normals)
    # else:
        # print(f"No points were sampled for {input_path}. Skipping.")


def main():
    """
    主函数，扫描目录，处理所有STEP文件。
    """
    parser = argparse.ArgumentParser(description='Scan a directory for .step files, sample points, and save them as .ply files.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the root input directory.')
    parser.add_argument('--num_points', type=int, required=True, help='Number of points to sample for each file.')
    parser.add_argument('--method', type=str, default='parametric', choices=['mesh', 'parametric'], help='Sampling method: "mesh" (faster, approximate) or "parametric" (slower, more accurate).')
    parser.add_argument('--with_normals', action='store_true', help='If set, sample surface normals along with points.')
    
    args = parser.parse_args()

    if args.method == 'parametric' and not PYTHON_OCC_AVAILABLE:
        print("Error: 'parametric' method requires python-occ-core, but it's not installed.")
        print("Please install it via: conda install -c conda-forge python-occ-core")
        return

    source_dir = os.path.join(args.input_dir, 'cad_step')
    target_dir = os.path.join(args.input_dir, 'pc_normal')

    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found at {source_dir}")
        return

    step_files = glob.glob(os.path.join(source_dir, '**', '*.step'), recursive=True)
    
    if not step_files:
        print(f"No .step files found in {source_dir}")
        return

    print(f"Found {len(step_files)} .step files to process.")

    # 使用多进程并行处理
    num_cores = 32
    print(f"Using {num_cores} cores for parallel processing.")

    # 创建一个处理函数，该函数接受一个文件路径并调用 process_file
    # functools.partial 用来固定 process_file 的部分参数
    process_func = partial(
        process_single_file_wrapper,
        source_dir=source_dir,
        target_dir=target_dir,
        num_points=args.num_points,
        method=args.method,
        with_normals=args.with_normals
    )

    with Pool(num_cores) as pool:
        # 使用 imap_unordered 以便在任务完成时立即更新进度条
        list(tqdm(pool.imap_unordered(process_func, step_files), total=len(step_files), desc="Processing STEP files"))

    print("Processing complete.")

def process_single_file_wrapper(step_file, source_dir, target_dir, num_points, method, with_normals):
    """
    包装器函数，用于从单个参数（step_file）调用 process_file，并包含异常处理。
    """
    try:
        relative_path = os.path.relpath(step_file, source_dir)
        output_filename = os.path.splitext(relative_path)[0] + '.ply'
        output_path = os.path.join(target_dir, output_filename)
        process_file(step_file, output_path, num_points, method, with_normals)
    except Exception as e:
        print(f"Failed to process {step_file}: {e}")

if __name__ == '__main__':
    main()