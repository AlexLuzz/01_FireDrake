#!/usr/bin/env python
"""
2.5D Geoelectrical modeling: div(sigma*grad(u)) - sigma*k^2*u = -I*delta(r-r_s)
Following Dey & Morrison (1979) for mixed boundary conditions
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from firedrake import *
from firedrake.pyplot import tricontourf, triplot
from scipy.special import k0
import gmsh

# Parameters
source_A, source_B = (-5.25, -3.75), (5.25, -3.75)
k_val, sigma_val = 1e-2, 1.0

def generate_ert_mesh(filename="ert_mesh.msh"):
    gmsh.initialize()
    gmsh.model.add("Geoelectrical_Domain")

    # --- Geometry Definition ---
    # Outer box corners
    p1 = gmsh.model.geo.addPoint(-10.0, -15.0, 0.0)
    p2 = gmsh.model.geo.addPoint( 10.0, -15.0, 0.0)
    p3 = gmsh.model.geo.addPoint( 10.0,   0.0, 0.0)
    p4 = gmsh.model.geo.addPoint(-10.0,   0.0, 0.0)

    # Electrodes (A and B)
    p_A = gmsh.model.geo.addPoint(-5.25, -3.75, 0.0)
    p_B = gmsh.model.geo.addPoint( 5.25, -3.75, 0.0)

    # Outer boundaries
    l1 = gmsh.model.geo.addLine(p1, p2) # Bottom
    l2 = gmsh.model.geo.addLine(p2, p3) # Right
    l3 = gmsh.model.geo.addLine(p3, p4) # Top (Surface)
    l4 = gmsh.model.geo.addLine(p4, p1) # Left

    # Surface definition
    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.geo.addPlaneSurface([loop])

    # 1. Synchronize the geometry kernel first
    gmsh.model.geo.synchronize()

    # 2. Use the correct mesh module namespace to embed the points
    gmsh.model.mesh.embed(0, [p_A, p_B], 2, surf)

    # --- Sizing Fields (The Secret Sauce for ERT) ---
    # 1. Compute distance to the electrode points
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "PointsList", [p_A, p_B])

    # 2. Set up a Threshold math field based on that distance
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.15)  # Coarsened from 0.04 to 15 cm at electrodes
    gmsh.model.mesh.field.setNumber(2, "SizeMax", 1.80)  # Coarsened from 1.20 to 1.8 m far away
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5)   # Fine zone radius
    gmsh.model.mesh.field.setNumber(2, "DistMax", 6.0)   # Transition zone radius

    # Activate the sizing field
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    # --- Physical Groups for Firedrake Boundary Markers ---
    # These integer IDs match your DirichletBC tags exactly
    gmsh.model.addPhysicalGroup(1, [l4], 1)        # Left boundary -> Tag 1
    gmsh.model.addPhysicalGroup(1, [l2], 2)        # Right boundary -> Tag 2
    gmsh.model.addPhysicalGroup(1, [l1], 3)        # Bottom boundary -> Tag 3
    gmsh.model.addPhysicalGroup(1, [l3], 4)        # Top boundary -> Tag 4
    gmsh.model.addPhysicalGroup(2, [surf], 1)      # Domain Interior

    # --- Mesh Generation ---
    gmsh.option.setNumber("Mesh.Algorithm", 6)     # Frontal-Delaunay (great for graded meshes)
    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()

# Generate it
#generate_ert_mesh("unstructured_ert.msh")

# Load the unstructured mesh
#mesh = Mesh("unstructured_ert.msh")

mesh = RectangleMesh(40, 30, 20.0, 15.0, quadrilateral=True)

# 2. Shift to center it on [-10, 10] x [-15, 0]
X = mesh.coordinates
with X.dat.vec as v:
    arr = v.array.reshape((-1, 2))
    arr[:, 0] -= 10.0
    arr[:, 1] -= 15.0

# 2. FIX: Define V ONCE and stick to it (CG1 is fine for this mesh density)
V = FunctionSpace(mesh, "CG", 2)

# Analytical solution following Dey & Morrison (1979)
def analytical_solution(V, src_A, src_B, k, sigma):
    v_coords = Function(VectorFunctionSpace(mesh, "CG", V.ufl_element().degree()))
    v_coords.interpolate(SpatialCoordinate(mesh))
    coords = v_coords.dat.data
    
    def potential(src):
        r_pos = np.maximum(np.sqrt((coords[:, 0] - src[0])**2 + (coords[:, 1] - src[1])**2), 1e-12)
        r_neg = np.maximum(np.sqrt((coords[:, 0] - src[0])**2 + (coords[:, 1] + src[1])**2), 1e-12)
        return (1.0 / (2.0 * np.pi * sigma)) * (k0(r_pos * k) + k0(r_neg * k))
    
    u_ana = Function(V)
    u_ana.dat.data[:] = potential(src_A) - potential(src_B)
    return u_ana

u_exact = analytical_solution(V, source_A, source_B, k_val, sigma_val)

# --- Variational Problem Setup ---
u = TrialFunction(V)
v = TestFunction(V)

sigma = Constant(sigma_val)
k = Constant(k_val)

a = sigma * inner(grad(u), grad(v)) * dx + sigma * k**2 * u * v * dx
L = Constant(0.0) * v * dx

# 3. FIX: Apply Dirichlet BCs to the truncated boundaries (left: 1, right: 2, bottom: 3)
# Keep the top (4) as an implicit natural Neumann BC (air interface)
bc = DirichletBC(V, u_exact, [1, 2, 3])

def apply_point_source(V, x0, value, b):
    mesh = V.mesh()
    W = VectorFunctionSpace(mesh, V.ufl_element().family(), V.ufl_element().degree())
    X = Function(W).interpolate(SpatialCoordinate(mesh))
    
    coords = X.dat.data_ro
    x0 = np.array(x0)

    distances = np.linalg.norm(coords - x0, axis=1)
    
    if len(distances) > 0:
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        
        tol = 1e-6
        if min_dist < tol:
            b.dat.data[min_idx] += value

# --- Assemble system with boundary conditions ---
A = assemble(a, bcs=[bc]) 
b = assemble(L, bcs=[bc])

# Apply Point sources directly to the internal nodes of the assembled vector
apply_point_source(V, source_A,  1.0, b)
apply_point_source(V, source_B, -1.0, b)

u_num = Function(V, name="Electrical_Potential")

# --- Solve ---
solve(A, u_num, b,
      solver_parameters={
          "ksp_type": "preonly",
          "pc_type": "lu"
      })

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cnt0 = tricontourf(u_num, axes=axes[0], levels=20, cmap='RdBu_r')
axes[0].set_title("Potential $u$ [V]")
plt.colorbar(cnt0, ax=axes[0], label='Potential [V]')

# 4. FIX: Use NumPy for absolute difference to bypass UFL projection artifacts
error_log = Function(V)
error_data = np.abs(u_num.dat.data_ro - u_exact.dat.data_ro)
error_log.dat.data[:] = np.log10(np.clip(error_data, 1e-12, None))

levels_log = np.arange(-7, 0, 1)
cnt1 = tricontourf(error_log, levels=levels_log, extend='both', cmap='Reds', axes=axes[1])
triplot(mesh, axes=axes[1], interior_kw={'edgecolor': 'black', 'linewidth': 0.3, 'alpha': 0.3})
axes[1].set_title("Error $|u_{exact} - u_{num}|$ [V]")

cbar = plt.colorbar(cnt1, ax=axes[1], ticks=levels_log)
cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"$10^{{{int(x)}}}$"))
cbar.set_label('Error [V]')

for ax in axes:
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-15, 0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

plt.tight_layout()
plt.savefig('firedrake_geoelectric.png', dpi=150, bbox_inches='tight')

# --- Error norms ---
L2_error = sqrt(assemble(inner(u_num - u_exact, u_num - u_exact) * dx))
H1_error = sqrt(assemble(inner(grad(u_num - u_exact), grad(u_num - u_exact)) * dx))

# Note: L2 error will still read as 'large' due to the numerical vs analytical 
print(f"L2 error: {L2_error:.6e}")
print(f"H1 error: {H1_error:.6e}")