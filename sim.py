import warp as wp
import torch
import warp.autograd
import numpy as np
import open3d as o3d
from tqdm import tqdm
from export_video import export_gif, export_mp4
from log import create_folder
from pbrt_renderer import PbrtRenderer
from config import to_real_array
import os
from argparse import ArgumentParser
from deepsdf import DeepSDFWithCode, device
from export_video import export_gif

integer = int
real = wp.float32
vec = wp.vec3
mat = wp.mat33
h = real(0.08)
damping = real(1e-2)
pcd_folder = "/mnt/data1/xiongxy/pcd_soft"

parser = ArgumentParser()
parser.add_argument("--name", "-n", required=True, type=str)
args = parser.parse_args()

#############################################################################################
# Initial setup
#############################################################################################
# Load point cloud
pcd_outer = o3d.io.read_point_cloud(pcd_folder + f"/{args.name}/point_cloud_downsampled.ply")
pcd_inner = o3d.io.read_point_cloud(pcd_folder + f"/{args.name}/{args.name}_inner.ply")
R = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
pcd_outer_np = np.asarray(pcd_outer.points)
pcd_inner_np = np.asarray(pcd_inner.points)
points_np = np.vstack([pcd_outer_np, pcd_inner_np])
delta = np.max(points_np, axis=0) * 1.5 * np.array([0, 1, 0])
points_np -= delta
points_torch = torch.from_numpy(points_np).to(device).float()
points_np += delta
n_points = points_np.shape[0]
points_np = points_np @ R + np.array([0., 1.2, 0.])
out_num = pcd_outer_np.shape[0]

sdf = DeepSDFWithCode().to(device)
try:
    min_loss_index = np.load("/mnt/data1/xiongxy/model_soft/" + args.name + "/min_loss_index.npy")
except:
    min_loss_index = 10000
sdf.load_state_dict(torch.load("/mnt/data1/xiongxy/model_soft/" + args.name + "/model_" + str(min_loss_index) +".pth"))

# Simulation parameters
frames = 6500
time_step = real(1.2e-4)

# Collision parameters
collision_penalty_stiffness = real(1e5)
collision_range = real(1e-3)

# Material properties
youngs_modulus = wp.array(shape=n_points, dtype=real, device="cuda")
poisson_ratio = wp.array(shape=n_points, dtype=real, device="cuda")
mu = wp.array(shape=n_points, dtype=real, device="cuda")
lam = wp.array(shape=n_points, dtype=real, device="cuda")

# Particle properties
mass = wp.array(shape=n_points, dtype=real, device="cuda")
rho = wp.array(shape=n_points, dtype=real, device="cuda")
volume = wp.array(shape=n_points, dtype=real, device="cuda")
free_points = wp.from_numpy(np.ones((n_points, 3)), dtype=vec, device="cuda")

# Particle state
position = wp.array(shape=(frames + 1, n_points), dtype=vec, device="cuda", requires_grad=True)
init_position = wp.from_numpy(points_np, dtype=vec, device="cuda")
init_position_float = wp.from_numpy(points_np, dtype=wp.vec3, device="cuda")
velocity = wp.array(shape=(frames + 1, n_points), dtype=vec, device="cuda", requires_grad=True)
# velocity.fill_(vec([0., -4., 0.]))
def_grad = wp.array(shape=(frames + 1, n_points), dtype=mat, device="cuda", requires_grad=True)

# Auxiliary fields
A_pq = wp.array(shape=(frames + 1, n_points), dtype=mat, device="cuda", requires_grad=True)

# Forces
external_forces = wp.array(shape=(n_points), dtype=vec, device="cuda", requires_grad=True)
elastic_forces = wp.array(shape=(frames + 1, n_points), dtype=vec, device="cuda", requires_grad=True)

# Optimization variables
x_np = sdf(points_torch).squeeze().detach().cpu().numpy()
x_np[:out_num] = np.clip(x_np[:out_num], 1., None)
x = wp.from_numpy(x_np, dtype=real, device="cuda", requires_grad=True)
ratio = wp.array(shape=n_points, dtype=real, device="cuda", requires_grad=True)

@wp.kernel
def compute_ratio(x: wp.array(dtype=real), ratio: wp.array(dtype=real)): # type: ignore
    i = wp.tid()
    ratio[i] = real(0.5) * wp.tanh(real(4.) * x[i]) + real(0.5)

l = wp.array(shape=(1), dtype=real, device="cuda", requires_grad=True)

target_position = wp.array(shape=n_points, dtype=vec, device="cuda")
target_velocity = wp.array(shape=n_points, dtype=vec, device="cuda")

grid_x = int(2 * (np.max(points_np[:, 0]) - np.min(points_np[:, 0])) / float(h) / 3)
grid_y = int(2 * (np.max(points_np[:, 1]) - np.min(points_np[:, 1])) / float(h) / 3)
grid_z = int(2 * (np.max(points_np[:, 2]) - np.min(points_np[:, 2])) / float(h) / 3)
grid = wp.HashGrid(grid_x, grid_y, grid_z, device="cuda")
grid.build(init_position_float, real(2.) * h)

#############################################################################################
# Computation functions
#############################################################################################
# Compute deformation gradient  
@wp.func
def W(xij: vec, h: real) -> real: # type: ignore
    q = wp.length(xij) / h
    ret = real(0.)
    if q < real(1.):
        ret = real(1.) / (real(wp.pi) * h * h * h) * (real(1.) - real(1.5) * q * q + real(0.75) * q * q * q)
    elif q >= real(1.) and q < real(2.):
        ret = real(1.) / (real(4.) * real(wp.pi) * h * h * h) * (real(2.) - q) * (real(2.) - q) * (real(2.) - q)
    return ret

@wp.func
def nabla_W(xij: vec, h: real) -> vec: # type: ignore
    q = wp.length(xij) / h
    ret = vec()
    if q < real(1.):
        ret = real(1.) / (real(wp.pi) * h * h * h) * (real(-3.) * xij / h / h + real(0.75) * real(3.) * q * xij / h / h)
    elif q >= real(1.) and q < real(2.):
        ret = real(1.) / (real(4.) * real(wp.pi) * h * h * h) * real(-3.) * (real(2.) - q) * (real(2.) - q) * xij / (q * h * h)
    return ret


@wp.kernel
def compute_v_i(grid: wp.uint64, position: wp.array(dtype=vec), mass: wp.array(dtype=real), rho: wp.array(dtype=real), volume: wp.array(dtype=real), h: real): # type: ignore
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x = position[i]
    r = real(0.0)

    neighbors = wp.hash_grid_query(grid, wp.vec3(x), 2. * float(h))
    # neighbors = range(position.shape[0])
    for index in neighbors:
        if index != i:
            r += mass[index] * W(x - position[index], h)
    rho[i] = r
    volume[i] = mass[i] / r


@wp.kernel
def compute_A_pq(grid: wp.uint64, position: wp.array2d(dtype=vec), init_position: wp.array(dtype=vec), mass: wp.array(dtype=real), A_pq: wp.array2d(dtype=mat), h: real, f: integer): # type: ignore
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x = position[f, i]
    x0 = init_position[i]
    a = mat()

    neighbors = wp.hash_grid_query(grid, wp.vec3(x0), 2. * float(h))
    for j in neighbors:
        if j != i:
            w = W(x0 - init_position[j], h)
            a += w * mass[j] * wp.outer(position[f, j] - x, init_position[j] - x0)
    A_pq[f, i] = a

@wp.func
def compute_R_i(A_pq: mat) -> mat: # type: ignore
    u = mat()
    sig = vec()
    v = mat()
    wp.svd3(A_pq, u, sig, v)
    return u @ wp.transpose(v)

@wp.kernel
def compute_nabla_u(grid: wp.uint64, position: wp.array2d(dtype=vec), init_position: wp.array(dtype=vec), volume: wp.array(dtype=real), A_pq: wp.array2d(dtype=mat), def_grad: wp.array2d(dtype=mat), h: real, f: integer): # type: ignore
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x0 = init_position[i]
    x = position[f, i]
    n_u = mat()

    R_pq_i = compute_R_i(A_pq[f, i])

    neighbors = wp.hash_grid_query(grid, wp.vec3(x0), 2. * float(h))
    for j in neighbors:
        if j != i:
            n_w = nabla_W(x0 - init_position[j], h)
            u_ji_bar = wp.transpose(R_pq_i) @ (position[f, j] - x) - (init_position[j] - x0)
            n_u += volume[j] * wp.outer(u_ji_bar, n_w)
    def_grad[f, i] = wp.identity(3, real) + wp.transpose(n_u)

# Compute elastic forces
@wp.func
def compute_sigma(def_grad: mat, mu: real, lam: real, ratio: real) -> mat: # type: ignore
    E = real(0.5) * (wp.transpose(def_grad) @ def_grad - wp.identity(3, real))
    s = (real(2.) * mu * E + lam * wp.trace(E) * wp.identity(3, real)) * (0.01 + ratio * 0.99)
    return s

@wp.kernel
def compute_elastic_forces(grid: wp.uint64, position: wp.array2d(dtype=vec), init_position: wp.array(dtype=vec), volume: wp.array(dtype=real), A_pq: wp.array2d(dtype=mat), def_grad: wp.array2d(dtype=mat), mu: wp.array(dtype=real), lam: wp.array(dtype=real), ratio: wp.array(dtype=real), elastic_forces: wp.array2d(dtype=vec), h: real, f: integer): # type: ignore
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x0 = init_position[i]
    force = vec()
    neighbors = wp.hash_grid_query(grid, wp.vec3(x0), 2. * float(h))
    R_pq_i = compute_R_i(A_pq[f, i])
    s_i = compute_sigma(def_grad[f, i], mu[i], lam[i], ratio[i])
    for j in neighbors:
        if j != i:
            s_j = compute_sigma(def_grad[f, j], mu[j], lam[j], ratio[j])
            R_pq_j = compute_R_i(A_pq[f, j])
            n_w = nabla_W(init_position[i] - init_position[j], h)
            f_ji = -volume[i] * def_grad[f, i] @ s_i @ (volume[j] * n_w)
            f_ij = volume[j] * def_grad[f, i] @ s_j @ (volume[i] * n_w)
            force += real(0.5) * (R_pq_j @ f_ij - R_pq_i @ f_ji)
    elastic_forces[f, i] = force

# Compute collision penalty
@wp.func
def compute_collision_penalty(position: vec) -> vec: # type: ignore
    collision_penalty = vec()
    if position[1] < collision_range:
        delta = collision_range - position[1]
        collision_penalty[1] = delta * delta * collision_penalty_stiffness
    return collision_penalty

# Simulation Step
@wp.kernel
def part_1(position: wp.array2d(dtype=vec), velocity: wp.array2d(dtype=vec), mass: wp.array(dtype=real), external_forces: wp.array(dtype=vec), elastic_forces: wp.array2d(dtype=vec), free_points: wp.array(dtype=vec), damping: real, time_step: real, f: integer): # type: ignore
    i = wp.tid()
    force = external_forces[i] + elastic_forces[f, i] - damping * velocity[f, i] + compute_collision_penalty(position[f, i])
    position[f + 1, i] = position[f, i] + wp.cw_mul(time_step * velocity[f, i] + real(.5) * time_step * time_step * force / mass[i], free_points[i])

@wp.kernel
def part_2(position: wp.array2d(dtype=vec), velocity: wp.array2d(dtype=vec), mass: wp.array(dtype=real), external_forces: wp.array(dtype=vec), elastic_forces: wp.array2d(dtype=vec), free_points: wp.array(dtype=vec), damping: real, time_step: real, f: integer): # type: ignore
    i = wp.tid()
    force_1 = external_forces[i] + elastic_forces[f, i] - damping * velocity[f, i] + compute_collision_penalty(position[f, i])
    force_2 = external_forces[i] + elastic_forces[f + 1, i] - damping * velocity[f, i] + compute_collision_penalty(position[f + 1, i])
    velocity[f + 1, i] = velocity[f, i] + wp.cw_mul(time_step * (force_1 + force_2) / (real(2.) * mass[i]), free_points[i])

@wp.kernel
def advance(position: wp.array2d(dtype=vec), velocity: wp.array2d(dtype=vec), mass: wp.array(dtype=real), external_forces: wp.array(dtype=vec), elastic_forces: wp.array2d(dtype=vec), free_points: wp.array(dtype=vec), damping: real, time_step: real, f: integer): # type: ignore
    i = wp.tid()
    force = external_forces[i] + elastic_forces[f, i] - damping * velocity[f, i] * mass[i]
    position[f + 1, i] = position[f, i] + wp.cw_mul(time_step * velocity[f, i], free_points[i])
    velocity[f + 1, i] = velocity[f, i] + wp.cw_mul(time_step * force / (mass[i]), free_points[i])



@wp.kernel
def startup(position: wp.array2d(dtype=vec), velocity: wp.array2d(dtype=vec), init_position: wp.array(dtype=vec)): # type: ignore
    i = wp.tid()
    position[0, i] = init_position[i]
    velocity[0, i] = vec()


@wp.kernel
def compute_loss(position: wp.array2d(dtype=vec), velocity: wp.array2d(dtype=vec), target_position: wp.array(dtype=vec), target_velocity: wp.array(dtype=vec), l: wp.array(dtype=real)): # type: ignore
    i = wp.tid()
    wp.atomic_add(l, 0, wp.length_sq(position[frames, i] - target_position[i]))
    wp.atomic_add(l, 0, wp.length_sq(velocity[frames, i] - target_velocity[i]))
        

#############################################################################################
# Control functions
#############################################################################################
def set_external_force(i: integer, f: vec): # type: ignore
    wp.copy(external_forces, wp.array(f, dtype=vec), dest_offset=i, count=1)

def set_all_external_force(f: vec): # type: ignore
    external_forces.fill_(f)

def set_dirichlet(i: integer, d: vec): # type: ignore
    wp.copy(free_points, wp.array(d, dtype=vec), dest_offset=i, count=1)

# def set_youngs_modulus(i: integer, E: real):
#     youngs_modulus[i] = E
#     mu[i] = E / (2 * (1 + poisson_ratio[i]))
#     lam[i] = E * poisson_ratio[i] / ((1 + poisson_ratio[i]) * (1 - 2 * poisson_ratio[i]))

@wp.kernel
def set_youngs_modulus(E: real, youngs_modulus: wp.array(dtype=real), poisson_ratio: wp.array(dtype=real), mu: wp.array(dtype=real), lam: wp.array(dtype=real)): # type: ignore
    i = wp.tid()
    youngs_modulus[i] = E
    mu[i] = E / (real(2.) * (real(1.) + poisson_ratio[i]))
    lam[i] = E * poisson_ratio[i] / ((real(1.) + poisson_ratio[i]) * (real(1.) - real(2.) * poisson_ratio[i]))

# def set_poisson_ratio(i: integer, nu: real):
#     poisson_ratio[i] = nu
#     mu[i] = youngs_modulus[i] / (2 * (1 + poisson_ratio[i]))
#     lam[i] = youngs_modulus[i] * poisson_ratio[i] / ((1 + poisson_ratio[i]) * (1 - 2 * poisson_ratio[i]))

@wp.kernel
def set_poisson_ratio(nu: real, youngs_modulus: wp.array(dtype=real), poisson_ratio: wp.array(dtype=real), mu: wp.array(dtype=real), lam: wp.array(dtype=real)): # type: ignore
    i = wp.tid()
    poisson_ratio[i] = nu
    mu[i] = youngs_modulus[i] / (real(2.) * (real(1.) + poisson_ratio[i]))
    lam[i] = youngs_modulus[i] * poisson_ratio[i] / ((real(1.) + poisson_ratio[i]) * (real(1.) - real(2.) * poisson_ratio[i]))

def set_mass(i: integer, m: real): # type: ignore
    wp.copy(mass, wp.array([m]), dest_offset=i, count=1)
    wp.launch(kernel=compute_v_i, dim=n_points, inputs=[grid.id, init_position, mass, rho, volume, h])

def set_mass(m: real): # type: ignore
    mass.fill_(m)
    wp.launch(kernel=compute_v_i, dim=n_points, inputs=[grid.id, init_position, mass, rho, volume, h])

def set_target():
    wp.copy(target_position, init_position)
    target_velocity.fill_(vec([0., 0., 0.]))


#############################################################################################
# Simulation and visualization
#############################################################################################
def visualize(f, image_name):
    pos_np = position.numpy()
    ratio_np = ratio.numpy()
    r = PbrtRenderer()
    eye = to_real_array([8, 1, 0])
    look_at = to_real_array([0, 0, 0])
    eye = look_at + 1 * (eye - look_at)
    r.set_camera(eye=eye, look_at=look_at, up=[0, 1, 0], fov=40)
    r.add_infinite_light({
        "rgb L": (1., 1., 1.)
    })
    for i in range(n_points):
        # print(point)
        r.add_sphere(pos_np[f, i], 0.007, ("diffuse", { "rgb reflectance": (ratio_np[i], 0.0, 1 - ratio_np[i]) }))
    r.add_plane([0., 0., 0.], [0., 1., 0.], 50., ("diffuse", { "rgb reflectance": (0.5, 0.5, 0.5) }))
    r.set_image(pixel_samples=128, file_name=image_name,
        resolution=[1000, 1000])
    r.render(use_gpu="PBRT_OPTIX7_PATH" in os.environ)

def diff_sim():
    tape = wp.Tape()
    with tape:
        wp.launch(kernel=compute_ratio, dim=n_points, inputs=[x, ratio])
        wp.launch(kernel=compute_A_pq, dim=n_points, inputs=[grid.id, position, init_position, mass, A_pq, h, 0])
        wp.launch(kernel=compute_nabla_u, dim=n_points, inputs=[grid.id, position, init_position, volume, A_pq, def_grad, h, 0])
        wp.launch(kernel=compute_elastic_forces, dim=n_points, inputs=[grid.id, position, init_position, volume, A_pq, def_grad, mu, lam, ratio, elastic_forces, h, 0])
        for f in tqdm(range(frames)):
            wp.launch(kernel=part_1, dim=n_points, inputs=[position, velocity, mass, external_forces, elastic_forces, free_points, damping, time_step, f])
            # wp.launch(kernel=advance, dim=n_points, inputs=[position, velocity, mass, external_forces, elastic_forces, free_points, damping, time_step, f])
            wp.launch(kernel=compute_A_pq, dim=n_points, inputs=[grid.id, position, init_position, mass, A_pq, h, f + 1])
            wp.launch(kernel=compute_nabla_u, dim=n_points, inputs=[grid.id, position, init_position, volume, A_pq, def_grad, h, f + 1])
            wp.launch(kernel=compute_elastic_forces, dim=n_points, inputs=[grid.id, position, init_position, volume, A_pq, def_grad, mu, lam, ratio, elastic_forces, h, f + 1])
            wp.launch(kernel=part_2, dim=n_points, inputs=[position, velocity, mass, external_forces, elastic_forces, free_points, damping, time_step, f])
        wp.launch(kernel=compute_loss, dim=n_points, inputs=[position, velocity, target_position, target_velocity, l])
    # tape.backward(l)

def main():
    set_all_external_force(vec([0., -4e-1, 0.]))
    wp.launch(kernel=set_youngs_modulus, dim=n_points, inputs=[1e7, youngs_modulus, poisson_ratio, mu, lam])
    wp.launch(kernel=set_poisson_ratio, dim=n_points, inputs=[0.4, youngs_modulus, poisson_ratio, mu, lam])
    set_mass(4e-2)
    # edge = np.where(points_np[:, 2] > 0.85)[0]
    # for i in edge:
    #     set_dirichlet(int(i), vec())
    # pull = np.where(points_np[:, 2] < 0.5)[0]
    # for i in pull:
    #     set_external_force(int(i), vec([0., 0., -1.]))
    set_target()
    wp.launch(kernel=startup, dim=n_points, inputs=[position, velocity, init_position])
    diff_sim()

    print(x.grad.numpy())
    # warp.autograd.gradcheck_tape(tape)

    render_folder = f"./render/{args.name}"
    create_folder(render_folder, exist_ok=True)
    for f in range(0, frames, 50):
        visualize(f, render_folder + "/sim_{:04d}.png".format(f))
    export_gif(render_folder, render_folder + "/sim.gif", 25, "sim_", ".png")

if __name__ == '__main__':
    main()