"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render rgbd
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
import imageio
from PIL import Image, ImageDraw

from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image


def load_rgbd_data(path="data/rgbd_data.pkl", image_size=512, duration=200,device = None, output_file="images/360_pointcloud_plant1.gif", ):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if device is None:
        device = get_device()

    # Unproject both depth images into point clouds
    points1, rgba1 = unproject_depth_image(torch.tensor(data['rgb1']), torch.tensor(data['mask1']), torch.tensor(data['depth1']), data['cameras1'])
    point_cloud1 = pytorch3d.structures.Pointclouds(points=[points1], features=[rgba1]).to(device)
    
    points2, rgba2 = unproject_depth_image(torch.tensor(data['rgb2']), torch.tensor(data['mask2']), torch.tensor(data['depth2']), data['cameras2'])
    point_cloud2 = pytorch3d.structures.Pointclouds(points=[points2], features=[rgba2]).to(device)

    # Concatenate the point clouds and color values
    points = torch.cat((points1, points2), dim=0)
    rgba = torch.cat((rgba1, rgba2), dim=0)
    point_cloud3 = pytorch3d.structures.Pointclouds(points=[points], features=[rgba]).to(device)

    renders = []
    for theta in range(0, 360, 10):
        R = torch.tensor([
            [np.cos(np.radians(theta)), 0.0, -np.sin(np.radians(theta))],
            [0.0, 1.0, 0.0],
            [np.sin(np.radians(theta)), 0.0, np.cos(np.radians(theta))]
        ])
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R.unsqueeze(0), T=[[0, 0, 6]], device=device)
        renderer = get_points_renderer(image_size=image_size, device=device)
        rend = renderer(point_cloud3, cameras=cameras)  # Change Here
        rend = rend[0, ..., :3].cpu().numpy()
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        images.append(np.array(image))
    imageio.mimsave(output_file, images, duration=duration, loop = 0)

    return rend


def render_bridge(
    point_cloud_path="data/bridge_pointcloud.npz",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend


def render_torus(image_size=256, num_samples=100, device=None): # change here
    """
    Renders a torus using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    R_ = 3.0  # Large radius of the torus # Change this
    r = 1.5  # Radius of the tube  # Change this

    Phi, Theta = torch.meshgrid(phi, theta)

    x = (R_ + r * torch.cos(Theta)) * torch.cos(Phi)
    y = (R_ + r * torch.cos(Theta)) * torch.sin(Phi)
    z = r * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    renders = []
    for theta in range(0, 360, 10):
        R = torch.tensor([
            [np.cos(np.radians(theta)), 0.0, -np.sin(np.radians(theta))],
            [0.0, 1.0, 0.0],
            [np.sin(np.radians(theta)), 0.0, np.cos(np.radians(theta))]
        ])
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R.unsqueeze(0), T=[[0, 0, 12]], device=device)
        renderer = get_points_renderer(image_size=image_size, device=device)
        rend = renderer(sphere_point_cloud, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy()
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        images.append(np.array(image))
    imageio.mimsave("images/parametric_torus_100.gif", images, duration=4, loop = 0)   # change here

    return rend


def render_torus_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()

    renderer = get_mesh_renderer(image_size=image_size)

    min_value = -1.1
    max_value = 1.1

    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    R_ = 0.5  # Large radius of the torus
    r = 0.2 # Radius of the tube

    voxels = ((((X ** 2 + Y ** 2) ** 0.5) - R_) ** 2) + (Z ** 2) - (r ** 2)
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )

    renders = []
    for theta in range(0, 360, 10):
        R = torch.tensor([
            [np.cos(np.radians(theta)), 0.0, -np.sin(np.radians(theta))],
            [0.0, 1.0, 0.0],
            [np.sin(np.radians(theta)), 0.0, np.cos(np.radians(theta))]
        ])
        T = torch.tensor([[0, 0, 4]])  # Move the camera to the side
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R.unsqueeze(0), T=T, fov=60, device=device)
        lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, 2.0]], device=device,)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)  # (N, H, W, 3)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        images.append(np.array(image))
    imageio.mimsave("images/implicit_torus.gif", images, duration=4, loop = 0)

    return rend

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="rgbd",
        choices=["rgbd","point_cloud", "parametric", "implicit"], 
    )
    parser.add_argument("--output_path", type=str, default="images/implicit_torus.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    if args.render == "rgbd":
        image = load_rgbd_data()
    elif args.render == "point_cloud":
        image = render_bridge(image_size=args.image_size)
    elif args.render == "parametric":
        image = render_torus(image_size=args.image_size, num_samples=args.num_samples)
    elif args.render == "implicit":
        image = render_torus_mesh(image_size=args.image_size)
    else:
        raise Exception("Did not understand {}".format(args.render))
    plt.imsave(args.output_path, image)
