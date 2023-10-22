"""
Sample code to render a tetrahedron.

Usage:
    python -m starter.tetrahedron_mesh 
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np
import imageio
from PIL import Image, ImageDraw
from math import sqrt

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def render_cow( 
    image_size=256, 
    num_frames=120,
    duration=200,
    color=[0, 1, 0.9],
    device=None,
    output_file="images/360_tetrahedron_render.gif",
):
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    vertices = torch.tensor([[sqrt(8/9),0,-1/3], [-sqrt(2/9),sqrt(2/3),-1/3], [-sqrt(2/9),  -sqrt(2/3),-1/3], [0,0,1]])
    faces = torch.tensor([[3, 0, 2], [3, 1, 0], [3, 2, 1], [0, 1, 2]])
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Initialize an empty list to store rendered images
    renders = []
    for theta in range(0, 360, 10):
        R = torch.tensor([
            [np.cos(np.radians(theta)), 0.0, -np.sin(np.radians(theta))],
            [0.0, 1.0, 0.0],
            [np.sin(np.radians(theta)), 0.0, np.cos(np.radians(theta))]
        ])
        T = torch.tensor([[0, 0, 3]])  # Move the camera to the side
        
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R.unsqueeze(0), T=T, fov=60, device=device)
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        images.append(np.array(image))
    imageio.mimsave(output_file, images, duration=duration, loop = 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--output_file", type=str, default="images/360_tetrahedron_render.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    render_cow(
        image_size=args.image_size,
        num_frames=args.num_frames,
        duration=args.duration,
        output_file=args.output_file,
    )