import taichi as ti
import numpy as np
import open3d as o3d
from tqdm import tqdm
import time
from export_video import export_gif, export_mp4
from utils import SvdDifferential, W, nabla_W
from options import real, integer, dim, h, damping, p, project_folder

ti.init(default_fp=real, default_ip=integer, arch=ti.gpu)

#############################################################################################
# Initial setup
#############################################################################################
# Load point cloud
# point_cloud = o3d.io.read_point_cloud(project_folder + "/.ply")
# points_np = np.asarray(point_cloud.points)
points = []
for i in range(40):
    for j in range(20):
        for k in range(10):
            points.append([i, j, k])
points_np = np.array(points) * 0.05
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
A_pq_grad = ti.Matrix.field(dim, dim, dtype=real, shape=(n_points, dim))
R_grad = ti.Matrix.field(dim, dim, dtype=real, shape=(n_points, dim))
nabla_u_grad = ti.Matrix.field(dim, dim, dtype=real, shape=(n_points, dim))
sigma = ti.Matrix.field(dim, dim, dtype=real, shape=(n_points,))

# Forces
external_forces = ti.Vector.field(dim, dtype=real, shape=(n_points,))
elastic_forces = ti.Vector.field(dim, dtype=real, shape=(n_points,))
damping_forces = ti.Vector.field(dim, dtype=real, shape=(n_points,))
pressure_forces = ti.Vector.field(dim, dtype=real, shape=(n_points,))

# Optimization variables
x = ti.field(dtype=real, shape=(n_points,), needs_grad=True)
inside = np.where((points_np[:, 2] - 0.2) * (points_np[:, 2] - 0.8) < 0)[0]
x_np = -10 * np.ones(n_points)
x_np[inside] = 10
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
    for i in range(n_points):
        volume_i[i] = 0
        rho_i[i] = 0
    for i, j in ti.ndrange(n_points, n_points):
        rho_i[i] += mass[j] * W(init_position[i] - init_position[j], h)
    for i in range(n_points):
        volume_i[i] = mass[i] / rho_i[i]

@ti.func
def compute_A_pq(h: real):
    for i in range(n_points):
        A_pq[i] = ti.Matrix.zero(real, dim, dim)
    for i, j in ti.ndrange(n_points, n_points):
        A_pq[i] += W(init_position[i] - init_position[j], h) * mass[j] * (position[j] - position[i]).outer_product(init_position[j] - init_position[i])

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
    for i in range(n_points):
        nabla_u[i] = ti.Matrix.zero(real, dim, dim)
    for i, j in ti.ndrange(n_points, n_points):
        u_ji_bar = R_i[i].inverse() @ (position[j] - position[i]) - (init_position[j] - init_position[i])
        nabla_u[i] += volume_i[j] * u_ji_bar.outer_product(nabla_W(init_position[i] - init_position[j], h))
    for i in range(n_points):
        def_grad[i] = ti.Matrix.identity(real, dim) + nabla_u[i].transpose()

# Compute gradient of deformation gradient
@ti.func
def compute_A_pg_grad(h: real):
    for i, d in ti.ndrange(n_points, dim):
        A_pq_grad[i, d] = ti.Matrix.zero(real, dim, dim)
    for i, j, d in ti.ndrange(n_points, n_points, dim):
        if ratio[i] < 1e-3: continue
        A_pq_grad[i, d] += -W(init_position[i] - init_position[j], h) * mass[j] * ti.Vector.unit(dim, d, real).outer_product(init_position[j] - init_position[i])

@ti.func
def compute_R_grad():
    for i, d in ti.ndrange(n_points, dim):
        if ratio[i] < 1e-3: continue
        dU, dS, dV = SvdDifferential(A_pq[i], U_i[i], S_i[i], V_i[i], A_pq_grad[i, d])
        R_grad[i, d] = dU @ V_i[i].transpose() + U_i[i] @ dV.transpose()

@ti.func
def compute_nabla_u_grad(h: real):
    for i, d in ti.ndrange(n_points, dim):
        nabla_u_grad[i, d] = ti.Matrix.zero(real, dim, dim)
    for i, j, d in ti.ndrange(n_points, n_points, dim):
        if ratio[i] < 1e-3: continue
        R_inverse = R_i[i].inverse()
        d_R_x = -R_inverse @ R_grad[i, d] @ R_inverse @ (position[j] - position[i]) - R_inverse @ ti.Vector.unit(dim, d, real)
        nabla_u_grad[i, d] += volume_i[j] * d_R_x.outer_product(nabla_W(init_position[i] - init_position[j], h))

# Compute elastic forces
@ti.func
def compute_stress_strain():
    for i in range(n_points):
        E = 0.5 * (def_grad[i] @ def_grad[i].transpose() - ti.Matrix.identity(real, dim))
        sigma[i] = 2 * mu[i] * E + lam[i] * E.trace() * ti.Matrix.identity(real, dim)

@ti.func
def compute_elastic_forces(h: real):
    compute_stress_strain()
    for i in range(n_points):
        elastic_forces[i] = ti.Vector.zero(real, dim)
    for i, j in ti.ndrange(n_points, n_points):
        f_ji = -volume_i[i] * (ti.Matrix.identity(real, dim) + nabla_u[i].transpose()) @ sigma[i] @ (volume_i[j] * nabla_W(init_position[i] - init_position[j], h))
        f_ij = -volume_i[j] * (ti.Matrix.identity(real, dim) + nabla_u[j].transpose()) @ sigma[j] @ (volume_i[i] * nabla_W(init_position[j] - init_position[i], h))
        elastic_forces[i] += 0.5 * (R_i[j] @ f_ij - R_i[i] @ f_ji) * (1 - ratio[i])

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
    for i, d in ti.ndrange(n_points, dim):
        if ratio[i] < 1e-3: continue
        pressure_forces[i][d] = def_grad[i].determinant() * (def_grad[i].inverse() @ nabla_u_grad[i, d]).trace() * p * ratio[i] * volume_i[i]

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
    external_forces[i] = f

@ti.kernel
def set_external_force(f: ti.math.vec3):
    for i in range(n_points):
        external_forces[i] = f

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
view_status = \
'''
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 1.0, 1.0, 2.0 ],
			"boundingbox_min" : [ -1.0, -1.0, 2.0 ],
			"field_of_view" : 60.0,
			"front" : [ 0, -0.93, -0.2 ],
			"lookat" : [ 1, 0, 0 ],
			"up" : [ 0, 0, 1 ],
			"zoom" : 0.5
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
'''
def visualize(points):
    render_folder = project_folder + "/render"
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for i, point in enumerate(points):
        time_stamp = time.time()
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point)
        vis.clear_geometries()
        vis.add_geometry(point_cloud)
        vis.set_view_status(view_status)
        vis.poll_events()
        vis.update_renderer()
        image_name = render_folder + f"/image_{i:04d}.png"
        vis.capture_screen_image(image_name)
    vis.destroy_window()
    export_mp4(render_folder, project_folder + "/render.mp4", 100, "image_", ".png")

def main():
    # set_external_force(ti.Vector([0., 0., -6e-3]))
    set_youngs_modulus(5e7)
    set_poisson_ratio(0.4)
    set_mass(1e-3)
    # print(mass.to_numpy())
    # print(volume_i.to_numpy())
    left_most = np.where(points_np[:, 0] <= 0.05)[0]
    # print(left_most)
    for i in left_most:
        set_dirichlet(int(i), ti.Vector([0., 0., 0.]))
    position_list = [position.to_numpy()]
    time_step = 1e-3
    n_steps = 200
    for _ in tqdm(range(n_steps)):
        for _ in range(2):
            forward(time_step / 2, h)
        position_list.append(position.to_numpy())
    visualize(position_list)
    

if __name__ == '__main__':
    main()
