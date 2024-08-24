import taichi as ti
import numpy as np
import open3d as o3d
from tqdm import tqdm
import time
from export_video import export_gif, export_mp4
from utils import SvdDifferential, W, nabla_W
from options import real, integer, dim, h, damping, p, project_folder
from log import create_folder
from pbrt_renderer import PbrtRenderer
from config import to_real_array
import os

ti.init(default_fp=real, default_ip=integer, arch=ti.gpu, device_memory_GB = 22, debug=True)

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
frames = 50

# Material properties
youngs_modulus = ti.field(dtype=real, shape=(n_points,))
poisson_ratio = ti.field(dtype=real, shape=(n_points,))
mu = ti.field(dtype=real, shape=(n_points,))
lam = ti.field(dtype=real, shape=(n_points,))

# Particle properties
mass = ti.field(dtype=real, shape=(n_points,))
rho_i = ti.field(dtype=real, shape=(n_points,))
volume_i = ti.field(dtype=real, shape=(n_points,))
free_points = ti.Vector.field(dim, dtype=real, shape=(n_points,))
free_points.from_numpy(np.ones((n_points, dim)))

# Particle state
position = ti.Vector.field(dim, dtype=real, shape=(frames + 1, n_points,), needs_grad=True)
init_position = ti.Vector.field(dim, dtype=real, shape=(n_points,))
init_position.from_numpy(points_np)
velocity = ti.Vector.field(dim, dtype=real, shape=(frames + 1, n_points,), needs_grad=True)
def_grad = ti.Matrix.field(dim, dim, dtype=real, shape=(frames + 1, n_points,), needs_grad=True)
pressure_energy = ti.field(dtype=real, shape=(), needs_grad=True)

# Auxiliary fields
A_pq = ti.Matrix.field(dim, dim, dtype=real, shape=(frames + 1, n_points,), needs_grad=True)
R_i = ti.Matrix.field(dim, dim, dtype=real, shape=(frames + 1, n_points,), needs_grad=True)
S_i = ti.Matrix.field(dim, dim, dtype=real, shape=(frames + 1, n_points,), needs_grad=True)
nabla_u = ti.Matrix.field(dim, dim, dtype=real, shape=(frames + 1, n_points,), needs_grad=True)
# A_pq_grad = ti.Matrix.field(dim, dim, dtype=real, shape=(n_points, n_points, dim))
# R_grad = ti.Matrix.field(dim, dim, dtype=real, shape=(n_points, n_points, dim))
# nabla_u_grad = ti.Matrix.field(dim, dim, dtype=real, shape=(n_points, n_points, dim))
sigma = ti.Matrix.field(dim, dim, dtype=real, shape=(frames + 1, n_points,), needs_grad=True)

# Forces
external_forces = ti.Vector.field(dim, dtype=real, shape=(n_points,), needs_grad=True)
elastic_forces = ti.Vector.field(dim, dtype=real, shape=(frames + 1, n_points,), needs_grad=True)
damping_forces = ti.Vector.field(dim, dtype=real, shape=(frames + 1, n_points,), needs_grad=True)
pressure_forces = ti.Vector.field(dim, dtype=real, shape=(n_points,), needs_grad=True)

# Optimization variables
x = ti.field(dtype=real, shape=(n_points,), needs_grad=True)

x_np = -10 * np.ones(n_points)
def part(point):
    return (point[1] + 0.1) ** 2 + point[0] ** 2 < 0.2 ** 2 and point[2] > 0 and point[2] < 0.85
for i in range(n_points):
    if (part(points_np[i])):
        x_np[i] = 10
x.from_numpy(x_np)

ratio = ti.field(dtype=real, shape=(n_points,), needs_grad=True)
@ti.kernel
def compute_ratio():
    for i in range(n_points):
        ratio[i] = 0.5 * ti.tanh(5 * x[i]) + 0.5

l = ti.field(dtype=real, shape=(), needs_grad=True)

target_position = ti.Vector.field(dim, dtype=real, shape=(n_points,))
target_velocity = ti.Vector.field(dim, dtype=real, shape=(n_points,))


#############################################################################################
# Computation functions
#############################################################################################
# Compute deformation gradient  
@ti.func
def compute_v_i(h: real):
    volume_i.fill(0)
    rho_i.fill(0)
    for i, j in ti.ndrange(n_points, n_points):
        rho_i[i] += mass[j] * W(init_position[i] - init_position[j], h)
    for i in range(n_points):
        volume_i[i] = mass[i] / rho_i[i]

@ti.kernel
def compute_A_pq(h: real, f: ti.i32):
    for i, j in ti.ndrange(n_points, n_points):
        w = W(init_position[i] - init_position[j], h)
        # if w == 0:
        #     continue
        A_pq[f, i] = w * mass[j] * (position[f, j] - position[f, i]).outer_product(init_position[j] - init_position[i])

@ti.kernel
def compute_R_i(f: ti.i32):
    for i in range(n_points):
        R_i[f, i], S_i[f, i] = ti.polar_decompose(A_pq[f, i])


@ti.kernel
def compute_nabla_u(h: real, f: ti.i32):
    for i in range(n_points):
        nabla_u[f, i] = ti.Matrix.zero(real, dim, dim)
    for i, j in ti.ndrange(n_points, n_points):
        r, s = ti.polar_decompose(A_pq[f, i])
        n_w = nabla_W(init_position[i] - init_position[j], h)
        u_ji_bar = r.inverse() @ (position[f, j] - position[f, i]) - (init_position[j] - init_position[i])
        nabla_u[f, i] += volume_i[j] * u_ji_bar.outer_product(n_w)
    for i in range(n_points):
        def_grad[f, i] = ti.Matrix.identity(real, dim) + nabla_u[f, i].transpose()


def output_nabla_u(h: real, f: ti.i32):
    compute_A_pq(h, f)
    # compute_R_i(f)
    compute_nabla_u(h, f)

# Compute gradient of deformation gradient
# @ti.func
# def compute_A_pg_grad(h: real):
#     # ti.deactivate_all_snodes(A_pq_grad)
#     A_pq_grad.fill(ti.Matrix.zero(real, dim, dim))
#     for i, j, d in ti.ndrange(n_points, n_points, dim):
#         # if ratio[i] < 1e-5:
#         #     continue
#         w = W(init_position[i] - init_position[j], h)
#         if w == 0 or i == j:
#             continue
#         A_pq_grad[i, i, d] += -w * mass[j] * ti.Vector.unit(dim, d, real).outer_product(init_position[j] - init_position[i])
#         A_pq_grad[i, j, d] += w * mass[j] * ti.Vector.unit(dim, d, real).outer_product(init_position[j] - init_position[i])
#         # print(A_pq_grad[i, j, d])

# @ti.func
# def compute_R_grad():
#     for i, j, d in ti.ndrange(n_points, n_points, dim):
#     # ti.deactivate_all_snodes(R_grad)
#     # for i, j, d in A_pq_grad:
#         # if ratio[i] < 1e-5 or A_pq_grad[i, j, d].norm() == 0:
#         #     continue
#         dU, dS, dV = SvdDifferential(A_pq[i], U_i[i], S_i[i], V_i[i], A_pq_grad[i, j, d])
#         R_grad[i, j, d] = dU @ V_i[i].transpose() + U_i[i] @ dV.transpose()

# @ti.func
# def compute_nabla_u_grad(h: real):
#     nabla_u_grad.fill(ti.Matrix.zero(real, dim, dim))
#     # ti.deactivate_all_snodes(nabla_u_grad)
#     for i, j, d in ti.ndrange(n_points, n_points, dim):
#     # for i, j, d in R_grad:
#         # if ratio[i] < 1e-5 or A_pq_grad[i, j, d].norm() == 0 or i == j:
#         #     continue
#         R_inverse = R_i[i].inverse()
#         d_R_x_i = -R_inverse @ R_grad[i, i, d] @ R_inverse @ (position[j] - position[i]) - R_inverse @ ti.Vector.unit(dim, d, real)
#         d_R_x_j = -R_inverse @ R_grad[i, j, d] @ R_inverse @ (position[j] - position[i]) + R_inverse @ ti.Vector.unit(dim, d, real)
#         nabla_u_grad[i, i, d] += volume_i[j] * d_R_x_i.outer_product(nabla_W(init_position[i] - init_position[j], h))
#         nabla_u_grad[i, j, d] += volume_i[j] * d_R_x_j.outer_product(nabla_W(init_position[i] - init_position[j], h))

# @ti.kernel
# def output_nabla_u_grad(h: real):
#     compute_A_pg_grad(h)
#     compute_R_grad()
#     compute_nabla_u_grad(h)

# Compute elastic forces
@ti.kernel
def compute_elastic_forces(h: real, f: ti.i32):
    for i in range(n_points):
        E = 0.5 * (def_grad[f, i].transpose() @ def_grad[f, i] - ti.Matrix.identity(real, dim))
        sigma[f, i] = (2 * mu[i] * E + lam[i] * E.trace() * ti.Matrix.identity(real, dim)) * (1 - ratio[i])
    for i in range(n_points):
        elastic_forces[f, i] = ti.Vector.zero(real, dim)
    for i, j in ti.ndrange(n_points, n_points):
        n_w = nabla_W(init_position[i] - init_position[j], h)
        # if n_w.norm() == 0:
        #     continue
        f_ji = -volume_i[i] * (ti.Matrix.identity(real, dim) + nabla_u[f, i].transpose()) @ sigma[f, i] @ (volume_i[j] * n_w)
        f_ij = volume_i[j] * (ti.Matrix.identity(real, dim) + nabla_u[f, j].transpose()) @ sigma[f, j] @ (volume_i[i] * n_w)
        elastic_forces[f, i] += 0.5 * (R_i[f, j] @ f_ij - R_i[f, i] @ f_ji)

# Compute damping forces
@ti.kernel
def compute_damping_forces(f: ti.i32):
    for i in range(n_points):
        damping_forces[f, i] = -damping * velocity[f, i]

# Compute pressure forces
# @ti.func
# def compute_pressure_forces(h: real):
#     compute_A_pg_grad(h)
#     compute_R_grad()
#     compute_nabla_u_grad(h)
#     pressure_forces.fill(ti.Vector.zero(real, dim))
#     for i, j, d in ti.ndrange(n_points, n_points, dim):
#         if ratio[i] < 1e-5 or A_pq_grad[i, j, d].norm() == 0 :
#             continue
#         pressure_forces[j][d] += def_grad[i].determinant() * (def_grad[i].inverse() @ nabla_u_grad[i, j, d].transpose()).trace() * p * ratio[i] * volume_i[i]

# Simulation Step
@ti.kernel
def advance(time_step: real, f: ti.i32):
    for i in range(n_points):
        force = external_forces[i] + elastic_forces[f, i] + damping_forces[f, i]
        velocity[f + 1, i] = time_step * force / mass[i] * free_points[i]
        position[f + 1, i] = position[f, i] + time_step * velocity[f + 1, i] * free_points[i]

@ti.ad.grad_replaced
def forward(time_step: real, h: real, f: ti.i32):
    compute_A_pq(h, f)
    compute_R_i(f)
    compute_nabla_u(h, f)
    compute_elastic_forces(h, f)
    compute_damping_forces(f)
    advance(time_step, f)


@ti.kernel
def startup():
    for i in range(n_points):
        position[0, i] = init_position[i]
        velocity[0, i] = ti.Vector.zero(real, dim)
    for i in range(1):
        l[None] = 0


@ti.kernel
def compute_loss():
    for i in range(n_points):
        l[None] += (position[frames, i] - target_position[i]).norm() ** 2
        l[None] += (velocity[frames, i] - target_velocity[i]).norm() ** 2

@ti.kernel
def clear_grad():
    for i in range(n_points):
        position.grad[i] = ti.Vector.zero(real, dim)
        velocity.grad[i] = ti.Vector.zero(real, dim)
        x.grad[i] = 0
        ratio.grad[i] = 0

#############################################################################################
# Control functions
#############################################################################################
@ti.kernel
def set_external_force(i: integer, f: ti.math.vec3):
    external_forces[i] = f # * (1 - ratio[i])

@ti.kernel
def set_all_external_force(f: ti.math.vec3):
    for i in range(n_points):
        external_forces[i] = f# * (1 - ratio[i])

@ti.kernel
def set_dirichlet(i: integer, d: ti.math.vec3):
    free_points[i] = d

@ti.kernel
def set_youngs_modulus(i: integer, E: real):
    youngs_modulus[i] = E
    mu[i] = E / (2 * (1 + poisson_ratio[i]))
    lam[i] = E * poisson_ratio[i] / ((1 + poisson_ratio[i]) * (1 - 2 * poisson_ratio[i]))

@ti.kernel
def set_youngs_modulus(E: real):
    for i in range(n_points):
        youngs_modulus[i] = E
        mu[i] = E / (2 * (1 + poisson_ratio[i]))
        lam[i] = E * poisson_ratio[i] / ((1 + poisson_ratio[i]) * (1 - 2 * poisson_ratio[i]))

@ti.kernel
def set_poisson_ratio(i: integer, nu: real):
    poisson_ratio[i] = nu
    mu[i] = youngs_modulus[i] / (2 * (1 + poisson_ratio[i]))
    lam[i] = youngs_modulus[i] * poisson_ratio[i] / ((1 + poisson_ratio[i]) * (1 - 2 * poisson_ratio[i]))

@ti.kernel
def set_poisson_ratio(nu: real):
    for i in range(n_points):
        poisson_ratio[i] = nu
        mu[i] = youngs_modulus[i] / (2 * (1 + poisson_ratio[i]))
        lam[i] = youngs_modulus[i] * poisson_ratio[i] / ((1 + poisson_ratio[i]) * (1 - 2 * poisson_ratio[i]))

@ti.kernel
def set_mass(i: integer, m: real):
    mass[i] = m
    compute_v_i(h)

@ti.kernel
def set_mass(m: real):
    for i in range(n_points):
        mass[i] = m
    compute_v_i(h)

@ti.kernel
def set_target():
    for i in range(n_points):
        target_position[i] = position[frames, i]
        target_velocity[i] = velocity[frames, i]


#############################################################################################
# Simulation and visualization
#############################################################################################
def visualize(points, image_name, color=False):
    r = PbrtRenderer()
    eye = to_real_array([4, 1, 0])
    look_at = to_real_array([0, 0, 0])
    eye = look_at + 1 * (eye - look_at)
    r.set_camera(eye=eye, look_at=look_at, up=[0, 1, 0], fov=40)
    r.add_infinite_light({
        "rgb L": (0.7, 0.7, 0.7)
    })
    r.add_spherical_area_light([30, 10, 40], 3, [1, 1, 1], 3e4)
    i = 0
    for point in points:
        # print(point)
        if point[2] < 0:
            r.add_sphere(point, 0.007, ("diffuse", { "rgb reflectance": (0.8, 0.0, 0.0) }))
        elif part(point):
            r.add_sphere(point, 0.007, ("diffuse", { "rgb reflectance": (0.0, 0.8, 0.0) }))
        elif point[2] > 0.85:
            r.add_sphere(point, 0.007, ("diffuse", { "rgb reflectance": (0.0, 0.0, 0.8) }))
        else:
            r.add_sphere(point, 0.007, ("diffuse", { "rgb reflectance": (0.0, 0.0, 0.0) }))
        i += 1
    r.set_image(pixel_samples=32, file_name=image_name,
        resolution=[1000, 1000])
    r.render(use_gpu="PBRT_OPTIX7_PATH" in os.environ)

forward_pos_list = []
forward_vel_list = []

def loss(time_step):
    compute_ratio()
    for i in tqdm(range(frames)):
        forward(time_step, h, i)
    compute_loss()


def main():
    # set_external_force(ti.Vector([0., 0., -5e-2]))
    set_youngs_modulus(1e5)
    set_poisson_ratio(0.4)
    set_mass(1e-2)
    edge = np.where(points_np[:, 2] > 0.85)[0]
    for i in edge:
        set_dirichlet(i, ti.Vector([0., 0., 0.]))
    a = np.zeros_like(points_np)
    pull = np.where(points_np[:, 2] < 0)[0]
    a[pull, 2] = -10
    for i in pull:
        set_external_force(int(i), ti.Vector([0., 0., -1.]))
    time_step = 2e-3 / 50

    startup()
    with ti.ad.Tape(loss=l, validation=True):
        loss(time_step)

    # visualize(position[frames].to_numpy(), project_folder + "/final.png")
    set_target()
    
    

if __name__ == '__main__':
    main()
    # export_mp4(project_folder + "/render", project_folder + "/render.mp4", 50, "image_", ".png")
    # set_youngs_modulus(1e7)
    # set_poisson_ratio(0.4)
    # set_mass(5e-2)
    # output_nabla_u(h)
    # ni = 0 # indices(6, 8, 3)
    # nj = 1 # indices(6, 8, 3)
    # d = 0
    # nabla_u_np_0 = nabla_u[ni].to_numpy()
    # output_nabla_u_grad(h)
    # # print(A_pq_grad)
    # print("Grad Ana:", nabla_u_grad[ni, nj, d].to_numpy())
    # @ti.kernel
    # def updata_position(i: integer, d: integer, delta: real):
    #     position[i][d] += delta
    # eps = 1e-8
    # updata_position(nj, d, eps)
    # output_nabla_u(h)
    # nabla_u_np_1 = nabla_u[ni].to_numpy()
    # updata_position(nj, d, -2 * eps)
    # output_nabla_u(h)
    # nabla_u_np_2 = nabla_u[ni].to_numpy()
    # print("Grad Num:", (nabla_u_np_1 - nabla_u_np_2) / (2 * eps))