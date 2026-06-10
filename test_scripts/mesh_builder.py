import gmsh
from firedrake import Mesh

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

    # Embed electrodes into the surface layout so mesh elements align to them
    gmsh.model.geo.mesh.embed(0, [p_A, p_B], 2, surf)
    gmsh.model.geo.synchronize()

    # --- Sizing Fields (The Secret Sauce for ERT) ---
    # 1. Compute distance to the electrode points
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "PointsList", [p_A, p_B])

    # 2. Set up a Threshold math field based on that distance
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.04)  # Size exactly at the electrode (4 cm)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", 1.20)  # Size far away from the electrode (1.2 m)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.15)  # Stay ultra-fine within 15 cm
    gmsh.model.mesh.field.setNumber(2, "DistMax", 5.0)   # Smoothly transition out to 5 meters

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
generate_ert_mesh("unstructured_ert.msh")

# Load the unstructured mesh
mesh = Mesh("unstructured_ert.msh")