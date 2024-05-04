import taichi as ti
import numpy as np
import open3d as o3d
from tqdm import tqdm
import time
from export_video import export_gif, export_mp4
from utils import SvdDifferential, W, nabla_W
from options import real, integer, dim, h, damping, p, project_folder
from log import create_folder

ti.init(default_fp=real, default_ip=integer, arch=ti.gpu, device_memory_GB = 22)

#############################################################################################
# Initial setup
#############################################################################################
# Load point cloud
# point_cloud = o3d.io.read_point_cloud(project_folder + "/.ply")
# points_np = np.asarray(point_cloud.points)
points = []
Nx = 20
Ny = 20
Nz = 10
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            points.append([i, j, k])
def indices(i, j, k):
    return Ny * Nz * i + Nz * j + k
points_np = np.array(points) * 0.01
n_points = points_np.shape[0]

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
position = ti.Vector.field(dim, dtype=real, shape=(n_points,), needs_grad=True)
init_position = ti.Vector.field(dim, dtype=real, shape=(n_points,))
@ti.kernel
def compute_init_com() -> ti.math.vec3:
    com = ti.Vector.zero(real, dim)
    for i in range(n_points):
        com += init_position[i]
    return com / n_points
init_com = compute_init_com()
position.from_numpy(points_np)
init_position.from_numpy(points_np)
velocity = ti.Vector.field(dim, dtype=real, shape=(n_points,))
velocity.from_numpy(np.zeros((n_points, dim)))
velocity_inter = ti.Vector.field(dim, dtype=real, shape=(n_points,))
def_grad = ti.Matrix.field(dim, dim, dtype=real, shape=(n_points,))
pressure_energy = ti.field(dtype=real, shape=(), needs_grad=True)

# Auxiliary fields
A_pq = ti.Matrix.field(dim, dim, dtype=real, shape=(n_points,))
R_i = ti.Matrix.field(dim, dim, dtype=real, shape=(n_points,))
U_i = ti.Matrix.field(dim, dim, dtype=real, shape=(n_points,))
S_i = ti.Matrix.field(dim, dim, dtype=real, shape=(n_points,))
V_i = ti.Matrix.field(dim, dim, dtype=real, shape=(n_points,))
nabla_u = ti.Matrix.field(dim, dim, dtype=real, shape=(n_points,))
A_pq_grad = ti.Matrix.field(dim, dim, dtype=real, shape=(n_points, n_points, dim))
R_grad = ti.Matrix.field(dim, dim, dtype=real, shape=(n_points, n_points, dim))
nabla_u_grad = ti.Matrix.field(dim, dim, dtype=real, shape=(n_points, n_points, dim))
# A_pq_grad = ti.Matrix.field(dim, dim, dtype=real)
# R_grad = ti.Matrix.field(dim, dim, dtype=real)
# nabla_u_grad = ti.Matrix.field(dim, dim, dtype=real)
# block_a = ti.root.pointer(ti.ijk, (n_points, n_points, dim))
# block_b = ti.root.pointer(ti.ijk, (n_points, n_points, dim))
# block_c = ti.root.pointer(ti.ijk, (n_points, n_points, dim))
# block_a.place(A_pq_grad)
# block_b.place(R_grad)
# block_c.place(nabla_u_grad)
sigma = ti.Matrix.field(dim, dim, dtype=real, shape=(n_points,))

# Forces
external_forces = ti.Vector.field(dim, dtype=real, shape=(n_points,))
elastic_forces = ti.Vector.field(dim, dtype=real, shape=(n_points,))
damping_forces = ti.Vector.field(dim, dtype=real, shape=(n_points,))
pressure_forces = ti.Vector.field(dim, dtype=real, shape=(n_points,))

# Optimization variables
x = ti.field(dtype=real, shape=(n_points,), needs_grad=True)
x_np = -10 * np.ones(n_points)
for i in range(2, Nx - 2):
    for j in range(2, Ny - 2):
        for k in range(2, Nz - 2):
            x_np[indices(i, j, k)] = 10
x.from_numpy(x_np)
ratio = ti.field(dtype=real, shape=(n_points,))
@ti.kernel
def compute_ratio():
    for i in range(n_points):
        ratio[i] = 0.5 * ti.tanh(5 * x[i]) + 0.5
compute_ratio()


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

@ti.func
def compute_A_pq(h: real):
    A_pq.fill(ti.Matrix.zero(real, dim, dim))
    for i, j in ti.ndrange(n_points, n_points):
        w = W(init_position[i] - init_position[j], h)
        if w == 0:
            continue
        A_pq[i] += w * mass[j] * (position[j] - position[i]).outer_product(init_position[j] - init_position[i])

@ti.func
def compute_R_i():
    for i in range(n_points):
        W, S, V = ti.svd(A_pq[i])
        R_i[i] = W @ V.transpose()
        U_i[i] = W
        S_i[i] = S
        V_i[i] = V

@ti.func
def compute_nabla_u(h: real):
    nabla_u.fill(ti.Matrix.zero(real, dim, dim))
    for i, j in ti.ndrange(n_points, n_points):
        n_w = nabla_W(init_position[i] - init_position[j], h)
        if n_w.norm() == 0:
            continue
        u_ji_bar = R_i[i].inverse() @ (position[j] - position[i]) - (init_position[j] - init_position[i])
        nabla_u[i] += volume_i[j] * u_ji_bar.outer_product(n_w)
    for i in range(n_points):
        def_grad[i] = ti.Matrix.identity(real, dim) + nabla_u[i].transpose()

@ti.kernel
def output_nabla_u(h: real):
    compute_A_pq(h)
    compute_R_i()
    compute_nabla_u(h)

# Compute gradient of deformation gradient
@ti.func
def compute_A_pg_grad(h: real):
    # ti.deactivate_all_snodes(A_pq_grad)
    A_pq_grad.fill(ti.Matrix.zero(real, dim, dim))
    for i, j, d in ti.ndrange(n_points, n_points, dim):
        if ratio[i] < 1e-5:
            continue
        w = W(init_position[i] - init_position[j], h)
        if w == 0 or i == j:
            continue
        A_pq_grad[i, i, d] += -w * mass[j] * ti.Vector.unit(dim, d, real).outer_product(init_position[j] - init_position[i])
        A_pq_grad[i, j, d] += w * mass[j] * ti.Vector.unit(dim, d, real).outer_product(init_position[j] - init_position[i])
        # print(A_pq_grad[i, j, d])

@ti.func
def compute_R_grad():
    for i, j, d in ti.ndrange(n_points, n_points, dim):
    # ti.deactivate_all_snodes(R_grad)
    # for i, j, d in A_pq_grad:
        if ratio[i] < 1e-5 or A_pq_grad[i, j, d].norm() == 0:
            continue
        dU, dS, dV = SvdDifferential(A_pq[i], U_i[i], S_i[i], V_i[i], A_pq_grad[i, j, d])
        R_grad[i, j, d] = dU @ V_i[i].transpose() + U_i[i] @ dV.transpose()

@ti.func
def compute_nabla_u_grad(h: real):
    nabla_u_grad.fill(ti.Matrix.zero(real, dim, dim))
    # ti.deactivate_all_snodes(nabla_u_grad)
    for i, j, d in ti.ndrange(n_points, n_points, dim):
    # for i, j, d in R_grad:
        if ratio[i] < 1e-5 or A_pq_grad[i, j, d].norm() == 0 or i == j:
            continue
        R_inverse = R_i[i].inverse()
        d_R_x_i = -R_inverse @ R_grad[i, i, d] @ R_inverse @ (position[j] - position[i]) - R_inverse @ ti.Vector.unit(dim, d, real)
        d_R_x_j = -R_inverse @ R_grad[i, j, d] @ R_inverse @ (position[j] - position[i]) + R_inverse @ ti.Vector.unit(dim, d, real)
        nabla_u_grad[i, i, d] += volume_i[j] * d_R_x_i.outer_product(nabla_W(init_position[i] - init_position[j], h))
        nabla_u_grad[i, j, d] += volume_i[j] * d_R_x_j.outer_product(nabla_W(init_position[i] - init_position[j], h))

@ti.kernel
def output_nabla_u_grad(h: real):
    compute_A_pg_grad(h)
    compute_R_grad()
    compute_nabla_u_grad(h)

# Compute elastic forces
@ti.func
def compute_stress_strain():
    for i in range(n_points):
        E = 0.5 * (def_grad[i].transpose() @ def_grad[i] - ti.Matrix.identity(real, dim))
        sigma[i] = (2 * mu[i] * E + lam[i] * E.trace() * ti.Matrix.identity(real, dim))# * (1 - ratio[i])
        # J = def_grad[i].determinant()
        # F_T_inv = def_grad[i].inverse().transpose()
        # sigma[i] = (mu[i] * (def_grad[i] - F_T_inv) + lam[i] * (J - 1) * J * F_T_inv) * def_grad[i].transpose() / J * (1 - ratio[i])

@ti.func
def compute_elastic_forces(h: real):
    compute_stress_strain()
    elastic_forces.fill(ti.Vector.zero(real, dim))
    for i, j in ti.ndrange(n_points, n_points):
        n_w = nabla_W(init_position[i] - init_position[j], h)
        if n_w.norm() == 0:
            continue
        f_ji = -volume_i[i] * (ti.Matrix.identity(real, dim) + nabla_u[i].transpose()) @ sigma[i] @ (volume_i[j] * n_w)
        f_ij = volume_i[j] * (ti.Matrix.identity(real, dim) + nabla_u[j].transpose()) @ sigma[j] @ (volume_i[i] * n_w)
        elastic_forces[i] += 0.5 * (R_i[j] @ f_ij - R_i[i] @ f_ji)

# Compute damping forces
@ti.func
def compute_damping_forces():
    for i in range(n_points):
        damping_forces[i] = -damping * velocity[i]

# Compute pressure forces
@ti.func
def compute_pressure_forces(h: real):
    compute_A_pg_grad(h)
    compute_R_grad()
    compute_nabla_u_grad(h)
    pressure_forces.fill(ti.Vector.zero(real, dim))
    for i, j, d in ti.ndrange(n_points, n_points, dim):
        if ratio[i] < 1e-5 or A_pq_grad[i, j, d].norm() == 0 :
            continue
        pressure_forces[j][d] += def_grad[i].determinant() * (def_grad[i].inverse() @ nabla_u_grad[i, j, d].transpose()).trace() * p * ratio[i] * volume_i[i]

@ti.func
def compute_forces(h: real):
    compute_A_pq(h)
    compute_R_i()
    compute_nabla_u(h)
    compute_elastic_forces(h)
    compute_damping_forces()
    compute_pressure_forces(h)

# Simulation Step
@ti.kernel
def forward(time_step: real, h: real):
    compute_forces(h)
    for i in range(n_points):
        force = external_forces[i] + elastic_forces[i] + damping_forces[i] + pressure_forces[i]
        velocity_inter[i] = velocity[i] + time_step * force / mass[i] / 2 * free_points[i]
        position[i] += time_step * velocity_inter[i] * free_points[i]
    compute_forces(h)
    for i in range(n_points):
        force = external_forces[i] + elastic_forces[i] + damping_forces[i] + pressure_forces[i]
        velocity[i] = velocity_inter[i] + time_step * force / mass[i] / 2 * free_points[i]


#############################################################################################
# Control functions
#############################################################################################
@ti.kernel
def set_external_force(i: integer, f: ti.math.vec3):
    external_forces[i] = f# * (1 - ratio[i])

@ti.kernel
def set_external_force(f: ti.math.vec3):
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


#############################################################################################
# Simulation and visualization
#############################################################################################
def visualize(points):
    render_folder = project_folder + "/render"
    create_folder(render_folder, exist_ok=True)
    vis = o3d.visualization.rendering.OffscreenRenderer(800, 600)
    vis.setup_camera(60., np.array([0.1, 0, -0.05]), np.array([0.1, 0.4, 0.1]), np.array([0, 0, 1]))
    material = o3d.visualization.rendering.MaterialRecord()
    material.base_color = np.array([0., 0., 0., 1.0])
    material.shader = "defaultLit"
    material.point_size = 5
    for i, point in enumerate(points):
        time_stamp = time.time()
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point)
        vis.scene.clear_geometry()
        vis.scene.add_geometry("pcd", point_cloud, material)
        image_name = render_folder + f"/image_{i:04d}.png"
        image = vis.render_to_image()
        o3d.io.write_image(image_name, image)
    export_mp4(render_folder, project_folder + "/render.mp4", 50, "image_", ".png")

def main():
    # set_external_force(ti.Vector([0., 0., -5e-2]))
    set_youngs_modulus(1e8)
    set_poisson_ratio(0.4)
    set_mass(1e-2)
    print(mass)
    print(rho_i)
    left_most = np.where(points_np[:, 0] <= 0.015)[0]
    right_most = np.where(points_np[:, 0] >= 0.185)[0]
    # print(left_most)
    for i in left_most:
        set_dirichlet(int(i), ti.Vector([0., 0., 0.]))
    # for i in right_most:
    #     set_dirichlet(int(i), ti.Vector([0., 0., 0.]))
    position_list = [position.to_numpy()]
    time_step = 5e-3
    n_steps = 300
    for i in tqdm(range(n_steps)):
        zoom = 50
        for _ in range(zoom):
            forward(time_step / zoom, h)
        if i % 4 == 0:
            position_list.append(position.to_numpy())
        print(pressure_forces[indices(2, 2, 2)])
        print(pressure_forces[indices(10, 10, 5)])
        # print(position[indices(3, 3, 1)])
    visualize(position_list)
    

if __name__ == '__main__':
    main()
    # export_mp4(project_folder + "/render", project_folder + "/render.mp4", 50, "image_", ".png")
    # set_youngs_modulus(1e7)
    # set_poisson_ratio(0.4)
    # set_mass(1e-4)
    # output_nabla_u(h)
    # ni = indices(10, 10, 5)
    # nj = indices(10, 10, 5)
    # d = 2
    # nabla_u_np_0 = nabla_u[ni].to_numpy()
    # output_nabla_u_grad(h)
    # # print(A_pq_grad)
    # print("Grad Ana:", nabla_u_grad[ni, nj, d].to_numpy())
    # @ti.kernel
    # def updata_position(i: integer, d: integer, delta: real):
    #     position[i][d] += delta
    # eps = 1e-9
    # updata_position(nj, d, eps)
    # output_nabla_u(h)
    # nabla_u_np_1 = nabla_u[ni].to_numpy()
    # updata_position(nj, d, -2 * eps)
    # output_nabla_u(h)
    # nabla_u_np_2 = nabla_u[ni].to_numpy()
    # print("Grad Num:", (nabla_u_np_1 - nabla_u_np_2) / (2 * eps))