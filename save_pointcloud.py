import os
import numpy as np
import open3d as o3d
import argparse


def loader(input_path):
    pointcloud = np.load(input_path)
    pred_colors = pointcloud['pred_colors']
    unseen_xyz = pointcloud['unseen_xyz']
    pred_occupy = pointcloud['pred_occupy']
    return pred_colors, unseen_xyz, pred_occupy


def writer(pred_occ, pred_colors, unseen, threshold, input_name):
    pos = pred_occ > threshold
    points = unseen[pos].reshape((-1, 3))
    features = pred_colors[pos].reshape((-1, 3))
    good_points = points[:, 0] != -100

    if good_points.sum() == 0:
        pass

    # Create a new point cloud with filtered points and features
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points[good_points])
    pc.colors = o3d.utility.Vector3dVector(features[good_points])

    output_path = os.path.join(os.getcwd(), 'output', input_name, 'pointclouds')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_filename_ply = f"{input_name}_{threshold}.ply"
    output_filename_pcd = f"{input_name}_{threshold}.pcd"


    o3d.io.write_point_cloud(
        os.path.join(output_path, output_filename_ply),
        pc,
        write_ascii=False,  # Set this to False for binary output
        print_progress=True
    )

    o3d.io.write_point_cloud(
        os.path.join(output_path, output_filename_pcd),
        pc,
        write_ascii=False,  # Set this to True for ASCII output
        print_progress=True
    )
    

def main(args):
    input_name = args.input
    input_path = os.path.join(os.getcwd(), 'output', input_name, f"{input_name}_recon_data.npz")
    pred_colors, unseen_xyz, pred_occupy = loader(input_path)
    unseen = unseen_xyz.squeeze()
    pred_occ = (pred_occupy).squeeze()
    writer(pred_occ, pred_colors, unseen, args.threshold, input_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='spyro', help='input image name, no extension')
    parser.add_argument('--threshold', type=float, default=0.2, help='threshold for occupancy')
    args = parser.parse_args()
    main(args)
