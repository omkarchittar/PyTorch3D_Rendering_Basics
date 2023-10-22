"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np

from starter.utils import get_device, get_mesh_renderer


def render_cow(
    cow_path="data/cow_with_axis.obj",
    image_size=256,

    # # Uncomment for transform1:
    R_relative=[[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],

    # # Uncomment for transform2:
    # R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    # T_relative=[0, 0, 3],

    # # Uncomment for transform3:
    # R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    # T_relative=[0.5, -0.5, 0],

    # Uncomment for transform4:
    # R_relative=[[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    # T_relative=[3, 0, 3],
    
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)

    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    T = R_relative @ torch.tensor([0.0, 0.0, 3.0]) + T_relative
    renderer = get_mesh_renderer(image_size=image_size)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.t().unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow_with_axis.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="images/transform1.jpg")
    args = parser.parse_args()

    plt.imsave(args.output_path, render_cow(cow_path=args.cow_path, image_size=args.image_size))
