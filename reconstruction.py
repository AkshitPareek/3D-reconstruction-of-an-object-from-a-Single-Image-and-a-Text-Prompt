
import os, sys
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "SegmentAnything"))
sys.path.append(os.path.join(os.getcwd(), "MCC"))

import argparse
import copy

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert
from tqdm import tqdm

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate

import supervision as sv

# segment anything
from SegmentAnything.segment_anything import build_sam, SamPredictor 
import matplotlib.pyplot as plt


# MCC
from pytorch3d.io.obj_io import load_obj

import MCC.main_mcc as main_mcc
import MCC.mcc_model as mcc_model
import MCC.util.misc as misc
from MCC.engine_mcc import prepare_data, generate_html

import mcubes
import trimesh
import matplotlib.colors as colors
import imageio


# Load MCC for 3D reconstruction
def load_mcc_model(occupancy_weight, rgb_weight, device, args):
    if device == "cuda":
        model = mcc_model.get_mcc_model(
            occupancy_weight=1.0,
            rgb_weight=0.01,
            args=args,
        ).cuda()
    else:
        model = mcc_model.get_mcc_model(
            occupancy_weight=1.0,
            rgb_weight=0.01,
            args=args,
        ).cpu()
    misc.load_model(args=args, model_without_ddp=model, optimizer=None, loss_scaler=None)

    return model


# Load Grounding DINO for annotation
def load_grounding_dino(CONFIG_PATH, WEIGHTS_PATH):
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    return model


# Load Segment Anything for segmentation
def create_sam(device):
    """Load the segment-anything model, fetching the model-file as necessary."""
    sam_checkpoint = 'checkpoints/sam_vit_h_4b8939.pth'
    sam = build_sam(checkpoint=sam_checkpoint)
    return sam.to(device=device)


# Predict the bounding boxes
def run_grounding_dino(IMAGE_PATH, model, TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD):
    image_source, image = load_image(IMAGE_PATH)

    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_THRESHOLD, 
        text_threshold=TEXT_THRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB
    return annotated_frame, boxes, logits, phrases

# Generate the segmentation mask
def run_sam(sam, rgb, boxes, device):
    sam_predictor = SamPredictor(sam)
    sam_predictor.set_image(rgb)
    H, W, _ = rgb.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, rgb.shape[:2]).to(device)
    masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
    return masks

# saving the mask file (optional)
def save_mask(image_mask_pil, IMAGE_NAME):
    output_folder = os.path.join(os.getcwd(), "output/obj_masks")
    filename = f"{os.path.splitext(IMAGE_NAME)[0]}_seg.png"

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the mask as a .png file
    image_mask_pil.save(f"{output_folder}/{filename}")
    mask_path = os.path.join(output_folder, filename)
    return mask_path

# Load ZoeDepth to estimate depth
def load_depth_model(device):
    depth_model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to(device).eval()
    return depth_model


# get principal point for camera intrinsics
def get_principal_point(boxes, H, W):
    scaled_boxes_cxcywh = boxes * torch.Tensor([W, H, W, H])
    center_x = scaled_boxes_cxcywh[:, 0]
    center_y = scaled_boxes_cxcywh[:, 1]

    return (center_x.item(), center_y.item())


# estimate camera intrinsics
def get_intrinsics(H,W, principal_point):
    """
    Intrinsics for a pinhole camera model.
    Assume fov of 55 degrees and central principal point
    of bounding box.
    """
    f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    # cx = 0.5 * W
    # cy = 0.5 * H
    cx, cy = principal_point
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]])


# generate a seen point cloud from depth map
def backproject_depth_to_pointcloud(depth, principal_point, rotation=np.eye(3), translation=np.zeros(3)):
    intrinsics = get_intrinsics(depth.shape[0], depth.shape[1], principal_point)
    # Get the depth map shape
    height, width = depth.shape

    # Create a matrix of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    uv_homogeneous = np.stack((u, v, np.ones_like(u)), axis=-1).reshape(-1, 3)

    # Invert the intrinsic matrix
    inv_intrinsics = np.linalg.inv(intrinsics)

    # Convert depth to the camera coordinate system
    points_cam_homogeneous = np.dot(uv_homogeneous, inv_intrinsics.T) * depth.flatten()[:, np.newaxis]

    # Convert to 3D homogeneous coordinates
    points_cam_homogeneous = np.concatenate((points_cam_homogeneous, np.ones((len(points_cam_homogeneous), 1))), axis=1)

    # Apply the rotation and translation to get the 3D point cloud in the world coordinate system
    extrinsics = np.hstack((rotation, translation[:, np.newaxis]))
    pointcloud = np.dot(points_cam_homogeneous, extrinsics.T)

    # Reshape the point cloud back to the original depth map shape
    pointcloud = pointcloud[:, :3].reshape(height, width, 3)

    return pointcloud


# save the point cloud as an .obj file(optional)
def save_point_cloud_to_obj(point_cloud, IMAGE_NAME):
    op_folder = os.path.join(os.getcwd(), "output/pcd_objects")
    op_filename = f"{os.path.splitext(IMAGE_NAME)[0]}.obj"

    if not os.path.exists(op_folder):
        os.makedirs(op_folder)
        
    output_file = f"{op_folder}/{op_filename}"
    with open(output_file, 'w') as f:
        for y in range(point_cloud.shape[0]):
            for x in range(point_cloud.shape[1]):
                point = point_cloud[y, x]
                f.write(f'v {point[0]} {point[1]} {point[2]}\n')

    return output_file
                

# project the point cloud to depth map(optional)
def point_cloud_to_depth_map(point_cloud, img_shape, principal_point):
    """
    Project a point cloud into a depth map.
    point_cloud: numpy array of shape (N, 3) with 3D coordinates in the camera frame
    K: intrinsic camera matrix
    img_shape: tuple with the shape of the depth map (height, width)
    """
    K = get_intrinsics(img_shape[0], img_shape[1], principal_point)
    # Project 3D points to 2D image coordinates
    points_2d = K @ point_cloud.T
    points_2d /= points_2d[2, :]
    points_2d = points_2d[:2, :].T

    # Round the 2D points to integers
    points_2d = np.round(points_2d).astype(int)

    # Filter out points outside the image dimensions
    valid_points = (0 <= points_2d[:, 0]) & (points_2d[:, 0] < img_shape[1]) & \
          (0 <= points_2d[:, 1]) & (points_2d[:, 1] < img_shape[0])
    points_2d = points_2d[valid_points]
    point_cloud = point_cloud[valid_points]

    # Create a depth map and fill in the depths at the corresponding 2D points
    depth_map = np.zeros(img_shape, dtype=np.float32)
    depth_map[points_2d[:, 1], points_2d[:, 0]] = point_cloud[:, 2]

    return depth_map


def visualize_depth_map(depth_map):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display the depth map as an image
    ax.imshow(depth_map, cmap='jet')
    ax.set_title('Depth Map')
    ax.axis('off')

    # Show the plot
    plt.show()


# 3D reconstruction
def run_viz(model, samples, device, args, prefix):
    model.eval()

    seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_images = prepare_data(
        samples, device, is_train=False, args=args, is_viz=True
    )
    pred_occupy = []
    pred_colors = []

    max_n_unseen_fwd = 2000

    model.cached_enc_feat = None
    num_passes = int(np.ceil(unseen_xyz.shape[1] / max_n_unseen_fwd))
    for p_idx in tqdm(range(num_passes)):
        p_start = p_idx     * max_n_unseen_fwd
        p_end = (p_idx + 1) * max_n_unseen_fwd
        cur_unseen_xyz = unseen_xyz[:, p_start:p_end]
        cur_unseen_rgb = unseen_rgb[:, p_start:p_end].zero_()
        cur_labels = labels[:, p_start:p_end].zero_()

        with torch.no_grad():
            _, pred = model(
                seen_images=seen_images,
                seen_xyz=seen_xyz,
                unseen_xyz=cur_unseen_xyz,
                unseen_rgb=cur_unseen_rgb,
                unseen_occupy=cur_labels,
                cache_enc=True,
                valid_seen_xyz=valid_seen_xyz,
            )
        if device == "cuda":
            pred_occupy.append(pred[..., 0].cuda())
        else:
            pred_occupy.append(pred[..., 0].cpu())
        if args.regress_color:
            pred_colors.append(pred[..., 1:].reshape((-1, 3)))
        else:
            pred_colors.append(
                (
                    torch.nn.Softmax(dim=2)(
                        pred[..., 1:].reshape((-1, 3, 256)) / args.temperature
                    ) * torch.linspace(0, 1, 256, device=pred.device)
                ).sum(axis=2)
            )
    with open(prefix + f"_{os.path.splitext(args.image_name)[0]}" +'.html', 'a') as f:
        generate_html(
            None,
            None, None,
            torch.cat(pred_occupy, dim=1),
            torch.cat(pred_colors, dim=0),
            unseen_xyz,
            f,
            gt_xyz=None,
            gt_rgb=None,
            mesh_xyz=None,
            score_thresholds=args.score_thresholds,
            pointcloud_marker_size=3,
        )


def pad_image(im, value):
    if im.shape[0] > im.shape[1]:
        diff = im.shape[0] - im.shape[1]
        return torch.cat([im, (torch.zeros((im.shape[0], diff, im.shape[2])) + value)], dim=1)
    else:
        diff = im.shape[1] - im.shape[0]
        return torch.cat([im, (torch.zeros((diff, im.shape[1], im.shape[2])) + value)], dim=0)


def normalize(seen_xyz):
    seen_xyz = seen_xyz / (seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].var(dim=0) ** 0.5).mean()
    seen_xyz = seen_xyz - seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].mean(axis=0)
    return seen_xyz


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    CONFIG_PATH = os.path.join(os.getcwd(), "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    WEIGHTS_PATH = os.path.join(os.getcwd(), "checkpoints/groundingdino_swint_ogc.pth")

    GroundingDINO_model = load_grounding_dino(CONFIG_PATH, WEIGHTS_PATH)

    sam = create_sam(device)

    depth_model = load_depth_model(device)

    recon_model = load_mcc_model(occupancy_weight=1.0,
        rgb_weight=0.01,
        device=device,
        args=args)

    IMAGE_NAME = args.image_name
    TEXT_PROMPT = args.caption

    IMAGE_PATH = os.path.join(os.getcwd(), "input", IMAGE_NAME)

    bgr = cv2.imread(IMAGE_PATH)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    seen_rgb = (torch.tensor(bgr).float() / 255)[..., [2, 1, 0]]
    H, W = seen_rgb.shape[:2]
    seen_rgb = torch.nn.functional.interpolate(
        seen_rgb.permute(2, 0, 1)[None],
        size=[H, W],
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)


    annotated_frame, boxes, logits, phrases = run_grounding_dino(IMAGE_PATH, GroundingDINO_model, TEXT_PROMPT, BOX_THRESHOLD=0.35, TEXT_THRESHOLD=0.25)
    
    principal_point = get_principal_point(boxes, H=rgb.shape[0], W=rgb.shape[1])

    masks = run_sam(sam, rgb, boxes, device)

    # image_mask_pil = Image.fromarray(masks[0][0].cpu().numpy())

    # mask_path = save_mask(image_mask_pil, IMAGE_NAME)

    if device == 'cuda':
        depth = depth_model.infer(seen_rgb.permute(2, 0, 1)[None].cuda())
    else:
        depth = depth_model.infer(seen_rgb.permute(2, 0, 1)[None].cpu())
    depth = depth[0].permute(1, 2, 0)
    depth = depth.cpu().detach().numpy().squeeze()

    point_cloud = backproject_depth_to_pointcloud(depth, principal_point)

    # pcd_path = save_point_cloud_to_obj(point_cloud, IMAGE_NAME)

    seen_xyz = point_cloud
    seen_xyz = torch.tensor(seen_xyz).float()

    seg = masks[0][0].cpu().numpy().astype(np.uint8)

    mask = torch.tensor(cv2.resize(seg, (W, H))).bool()

    seen_xyz[~mask] = float('inf')

    seen_xyz = normalize(seen_xyz)

    bottom, right = mask.nonzero().max(dim=0)[0]
    top, left = mask.nonzero().min(dim=0)[0]

    bottom = bottom + 40
    right = right + 40
    top = max(top - 40, 0)
    left = max(left - 40, 0)

    seen_xyz = seen_xyz[top:bottom+1, left:right+1]
    seen_rgb = seen_rgb[top:bottom+1, left:right+1]

    seen_xyz = pad_image(seen_xyz, float('inf'))
    seen_rgb = pad_image(seen_rgb, 0)

    seen_rgb = torch.nn.functional.interpolate(
        seen_rgb.permute(2, 0, 1)[None],
        size=[800, 800],
        mode="bilinear",
        align_corners=False,
    )

    seen_xyz = torch.nn.functional.interpolate(
        seen_xyz.permute(2, 0, 1)[None],
        size=[112, 112],
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)

    samples = [
        [seen_xyz, seen_rgb],
        [torch.zeros((20000, 3)), torch.zeros((20000, 3))],
    ]
    run_viz(recon_model, samples, device, args, prefix=args.output)




if __name__ == '__main__':
    parser = main_mcc.get_args_parser()
    parser.add_argument('--image_name', default='spyro.jpg', type=str, help='input image name')
    parser.add_argument('--point_cloud', type=str, help='input obj file')
    parser.add_argument('--seg', type=str, help='input mask file')
    parser.add_argument('--caption', default='a toy', type=str, help='input text prompt')
    parser.add_argument('--output', default='output/3D', type=str, help='output path')
    parser.add_argument('--granularity', default=0.05, type=float, help='output granularity')
    parser.add_argument('--score_thresholds', default=[0.1, 0.2, 0.3, 0.4, 0.5], type=float, nargs='+', help='score thresholds')
    parser.add_argument('--temperature', default=0.1, type=float, help='temperature for color prediction.')
    parser.add_argument('--checkpoint', default='MCC/co3dv2_all_categories.pth', type=str, help='model checkpoint')

    parser.set_defaults(eval=True)

    args = parser.parse_args()
    args.resume = args.checkpoint
    args.viz_granularity = args.granularity
    main(args)





