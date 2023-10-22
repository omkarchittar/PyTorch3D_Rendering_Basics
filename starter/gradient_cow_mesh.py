"""
Sample code to render a cow.

Usage:
    python -m starter.gradient_cow_mesh --image_size 256 --output_path images/gradient_cow_render.jpg
"""

import argparse
import matplotlib.pyplot as plt
import pytorch3d
import torch
import imageio

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh

def render_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    if device is None:
        device = get_device()

    renderer = get_mesh_renderer(image_size=image_size)

    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    z_max = vertices[0, :, 2].max()
    z_min= vertices[0, :, 2].min()
    z = vertices[0, :, 2]
    alpha = (z - z_min) / (z_max - z_min)

    color1 = [1, 0, 1]
    color2 = [1, 1, 0]

    colors = alpha.view(-1, 1) * torch.tensor(color2).view(1, 1, 3) + (1 - alpha.view(-1, 1)) * torch.tensor(color1).view(1, 1, 3)

    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(colors)  # (1, N_v, 3)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 1]]), fov=60, device=device)  # Lowering the camera (T=[0, 0, 1])
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    output_path = "images/gradient_cow_render.gif"
    num_frames = 60
    fps = 15

    # Create views for the 360-degree rotation around the Y-axis
    elevations = torch.tensor(30.0, device=device)  # Constant elevation angle
    azimuths = torch.linspace(0, 360, num_frames, device=device)
    images = []
    for azimuth in azimuths:
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=3.0,
            elev=elevations,
            azim=azimuth
        )
        cameras.R = R.to(device)
        cameras.T = T.to(device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (B, H, W, 4) -> (H, W, 3)
        images.append((255 * rend).astype('uint8'))
    imageio.mimsave(output_path, images, duration=200, loop = 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/gradient_cow_render.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    render_cow(
        cow_path=args.cow_path,
        image_size=args.image_size
    )


