import warp as wp
import warp.autograd
import numpy as np
import open3d as o3d
from tqdm import tqdm
import time
from export_video import export_gif, export_mp4
from log import create_folder
from pbrt_renderer import PbrtRenderer
from config import to_real_array
import os

integer = int
real = wp.float32
vec = wp.vec3
mat = wp.mat33
h = real(0.1)
damping = real(1)
project_folder = "./pcd"

#############################################################################################
# Initial setup
#############################################################################################
# Load point cloud
pcd_outer = o3d.io.read_point_cloud(project_folder + "/spot_outer.ply")
pcd_inner = o3d.io.read_point_cloud(project_folder + "/spot_inner.ply")
pcd_outer_np = np.asarray(pcd_outer.points)
pcd_inner_np = np.asarray(pcd_inner.points)
points_np = np.vstack([pcd_outer_np, pcd_inner_np])
n_points = points_np.shape[0]

# Simulation parameters
frames = 1000
time_step = real(4e-4)

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
def_grad = wp.array(shape=(frames + 1, n_points), dtype=mat, device="cuda", requires_grad=True)

# Auxiliary fields
A_pq = wp.array(shape=(frames + 1, n_points), dtype=mat, device="cuda", requires_grad=True)
R_pq = wp.array(shape=(frames + 1, n_points), dtype=mat, device="cuda", requires_grad=True)
sigma = wp.array(shape=(frames + 1, n_points), dtype=mat, device="cuda", requires_grad=True)

# Forces
external_forces = wp.array(shape=(n_points), dtype=vec, device="cuda", requires_grad=True)
elastic_forces = wp.array(shape=(frames + 1, n_points), dtype=vec, device="cuda", requires_grad=True)

# Optimization variables
x = wp.array(shape=n_points, dtype=real, device="cuda", requires_grad=True)
ratio = wp.array(shape=n_points, dtype=real, device="cuda", requires_grad=True)

x.fill_(real(-1.0))

@wp.kernel
def compute_ratio(x: wp.array(dtype=real), ratio: wp.array(dtype=real)): # type: ignore
    i = wp.tid()
    ratio[i] = real(0.5) * wp.tanh(real(3.) * x[i]) + real(0.5)

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
def W(xij: vec, h: real) -> real:
    q = wp.length(xij) / h
    ret = real(0.)
    if q < real(1.):
        ret = real(1.) / (real(wp.pi) * h * h * h) * (real(1.) - real(1.5) * q * q + real(0.75) * q * q * q)
    elif q >= real(1.) and q < real(2.):
        ret = real(1.) / (real(4.) * real(wp.pi) * h * h * h) * (real(2.) - q) * (real(2.) - q) * (real(2.) - q)
    return ret

@wp.func
def nabla_W(xij: vec, h: real) -> vec:
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

@wp.kernel
def compute_R_i(A_pq: wp.array2d(dtype=mat), R_pq: wp.array2d(dtype=mat), f: integer): # type: ignore
    i = wp.tid()
    u = mat()
    sig = vec()
    v = mat()
    wp.svd3(A_pq[f, i], u, sig, v)
    R_pq[f, i] = u @ wp.transpose(v)

@wp.kernel
def compute_nabla_u(grid: wp.uint64, position: wp.array2d(dtype=vec), init_position: wp.array(dtype=vec), volume: wp.array(dtype=real), R_pq: wp.array2d(dtype=mat), def_grad: wp.array2d(dtype=mat), h: real, f: integer): # type: ignore
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x0 = init_position[i]
    x = position[f, i]
    n_u = mat()

    neighbors = wp.hash_grid_query(grid, wp.vec3(x0), 2. * float(h))
    for j in neighbors:
        if j != i:
            n_w = nabla_W(x0 - init_position[j], h)
            u_ji_bar = wp.inverse(R_pq[f, i]) @ (position[f, j] - x) - (init_position[j] - x0)
            n_u += volume[j] * wp.outer(u_ji_bar, n_w)
    def_grad[f, i] = wp.identity(3, real) + wp.transpose(n_u)

# Compute elastic forces
@wp.kernel
def compute_sigma(grid: wp.uint64, position: wp.array2d(dtype=vec), init_position: wp.array(dtype=vec), volume: wp.array(dtype=real), R_pq: wp.array2d(dtype=mat), def_grad: wp.array2d(dtype=mat), mu: wp.array(dtype=real), lam: wp.array(dtype=real), ratio: wp.array(dtype=real), sigma: wp.array2d(dtype=mat), h: real, f: integer): # type: ignore
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x0 = init_position[i]
    E = real(0.5) * (wp.transpose(def_grad[f, i]) @ def_grad[f, i] - wp.identity(3, real))
    s = (real(2.) * mu[i] * E + lam[i] * wp.trace(E) * wp.identity(3, real)) * (real(1.) - ratio[i])
    sigma[f, i] = s

@wp.kernel
def compute_elastic_forces(grid: wp.uint64, position: wp.array2d(dtype=vec), init_position: wp.array(dtype=vec), volume: wp.array(dtype=real), R_pq: wp.array2d(dtype=mat), def_grad: wp.array2d(dtype=mat), mu: wp.array(dtype=real), lam: wp.array(dtype=real), ratio: wp.array(dtype=real), sigma: wp.array2d(dtype=mat), elastic_forces: wp.array2d(dtype=vec), h: real, f: integer): # type: ignore
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x0 = init_position[i]
    force = vec()
    neighbors = wp.hash_grid_query(grid, wp.vec3(x0), 2. * float(h))
    for j in neighbors:
        if j != i:
            n_w = nabla_W(init_position[i] - init_position[j], h)
            f_ji = -volume[i] * def_grad[f, i] @ sigma[f, i] @ (volume[j] * n_w)
            f_ij = volume[j] * def_grad[f, i] @ sigma[f, j] @ (volume[i] * n_w)
            force += real(0.5) * (R_pq[f, j] @ f_ij - R_pq[f, i] @ f_ji)
    elastic_forces[f, i] = force

# Simulation Step
@wp.kernel
def part_1(position: wp.array2d(dtype=vec), velocity: wp.array2d(dtype=vec), mass: wp.array(dtype=real), external_forces: wp.array(dtype=vec), elastic_forces: wp.array2d(dtype=vec), free_points: wp.array(dtype=vec), damping: real, time_step: real, f: integer): # type: ignore
    i = wp.tid()
    force = external_forces[i] + elastic_forces[f, i] - damping * velocity[f, i]
    position[f + 1, i] = position[f, i] + wp.cw_mul(time_step * velocity[f, i] + real(.5) * time_step * time_step * force / mass[i], free_points[i])

@wp.kernel
def part_2(position: wp.array2d(dtype=vec), velocity: wp.array2d(dtype=vec), mass: wp.array(dtype=real), external_forces: wp.array(dtype=vec), elastic_forces: wp.array2d(dtype=vec), free_points: wp.array(dtype=vec), damping: real, time_step: real, f: integer): # type: ignore
    i = wp.tid()
    force_1 = external_forces[i] + elastic_forces[f, i] - damping * velocity[f, i]
    force_2 = external_forces[i] + elastic_forces[f + 1, i] - damping * velocity[f, i]
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
def set_external_force(i: integer, f: vec):
    wp.copy(external_forces, wp.array(f, dtype=vec), dest_offset=i, count=1)

def set_all_external_force(f: vec):
    external_forces.fill_(f)

def set_dirichlet(i: integer, d: vec):
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

def set_mass(i: integer, m: real):
    wp.copy(mass, wp.array([m]), dest_offset=i, count=1)
    wp.launch(kernel=compute_v_i, dim=n_points, inputs=[grid.id, init_position, mass, rho, volume, h])

def set_mass(m: real):
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
    r = PbrtRenderer()
    eye = to_real_array([4, 1, 0])
    look_at = to_real_array([0, 0, 0])
    eye = look_at + 1 * (eye - look_at)
    r.set_camera(eye=eye, look_at=look_at, up=[0, 1, 0], fov=40)
    r.add_infinite_light({
        "rgb L": (0.7, 0.7, 0.7)
    })
    r.add_spherical_area_light([30, 10, 40], 3, [1, 1, 1], 3e4)
    for i in range(n_points):
        # print(point)
        r.add_sphere(pos_np[f, i], 0.007, ("diffuse", { "rgb reflectance": (0.0, 0.0, 0.0) }))
    r.set_image(pixel_samples=32, file_name=image_name,
        resolution=[1000, 1000])
    r.render(use_gpu="PBRT_OPTIX7_PATH" in os.environ)


def main():
    # set_external_force(ti.Vector([0., 0., -5e-2]))
    wp.launch(kernel=set_youngs_modulus, dim=n_points, inputs=[1e5, youngs_modulus, poisson_ratio, mu, lam])
    wp.launch(kernel=set_poisson_ratio, dim=n_points, inputs=[0.4, youngs_modulus, poisson_ratio, mu, lam])
    set_mass(1e-2)
    edge = np.where(points_np[:, 2] > 0.85)[0]
    for i in edge:
        set_dirichlet(int(i), vec())
    pull = np.where(points_np[:, 2] < 0.5)[0]
    for i in pull:
        set_external_force(int(i), vec([0., 0., -1.]))
    set_target()
    wp.launch(kernel=startup, dim=n_points, inputs=[position, velocity, init_position])

    tape = wp.Tape()
    with tape:
        wp.launch(kernel=compute_ratio, dim=n_points, inputs=[x, ratio])
        wp.launch(kernel=compute_A_pq, dim=n_points, inputs=[grid.id, position, init_position, mass, A_pq, h, 0])
        wp.launch(kernel=compute_R_i, dim=n_points, inputs=[A_pq, R_pq, 0])
        wp.launch(kernel=compute_nabla_u, dim=n_points, inputs=[grid.id, position, init_position, volume, R_pq, def_grad, h, 0])
        wp.launch(kernel=compute_sigma, dim=n_points, inputs=[grid.id, position, init_position, volume, R_pq, def_grad, mu, lam, ratio, sigma, h, 0])
        wp.launch(kernel=compute_elastic_forces, dim=n_points, inputs=[grid.id, position, init_position, volume, R_pq, def_grad, mu, lam, ratio, sigma, elastic_forces, h, 0])
        for f in tqdm(range(frames)):
            wp.launch(kernel=part_1, dim=n_points, inputs=[position, velocity, mass, external_forces, elastic_forces, free_points, damping, time_step, f])
            # wp.launch(kernel=advance, dim=n_points, inputs=[position, velocity, mass, external_forces, elastic_forces, free_points, damping, time_step, f])
            wp.launch(kernel=compute_A_pq, dim=n_points, inputs=[grid.id, position, init_position, mass, A_pq, h, f + 1])
            wp.launch(kernel=compute_R_i, dim=n_points, inputs=[A_pq, R_pq, f + 1])
            wp.launch(kernel=compute_nabla_u, dim=n_points, inputs=[grid.id, position, init_position, volume, R_pq, def_grad, h, f + 1])
            wp.launch(kernel=compute_sigma, dim=n_points, inputs=[grid.id, position, init_position, volume, R_pq, def_grad, mu, lam, ratio, sigma, h, f + 1])
            wp.launch(kernel=compute_elastic_forces, dim=n_points, inputs=[grid.id, position, init_position, volume, R_pq, def_grad, mu, lam, ratio, sigma, elastic_forces, h, f + 1])
            wp.launch(kernel=part_2, dim=n_points, inputs=[position, velocity, mass, external_forces, elastic_forces, free_points, damping, time_step, f])
        wp.launch(kernel=compute_loss, dim=n_points, inputs=[position, velocity, target_position, target_velocity, l])

    tape.backward(l)
    print(x.grad.numpy())
    # warp.autograd.gradcheck_tape(tape)

    visualize(0, project_folder + "/start.png")
    visualize(frames, project_folder + f"/final.png")

    # print(x.grad)
    # i = 400
    # grad = x.grad[i]

    # eps = 1e-5
    # startup()
    # x[i] += eps
    # loss(time_step)
    # l_1 = l[None]

    # startup()
    # x[i] -= 2 * eps
    # loss(time_step)
    # l_2 = l[None]

    # print("Grad Ana:", grad)
    # print("Grad Num:", (l_1 - l_2) / (2 * eps))

    # set_target()

if __name__ == '__main__':
    main()