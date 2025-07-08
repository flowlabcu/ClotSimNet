import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

'''
For a given number of voids in the x direction and the radius of the voids, compute the 
positions of the voids and the interstitial space between them. The space should be greater than 
the mesh maximum size.
'''


def write_gmsh(filename, positions, radius, domain, h):
    # # --- Gmsh .geo script output ---
    with open(f'{filename}', 'w') as f:
        f.write('SetFactory("OpenCASCADE");\n')
        f.write('Mesh.MshFileVersion = 2.0;\n\n')

        # Box
        f.write(f'Box(1) = {{0, 0, 0, {domain[0]}, {domain[1]}, {domain[2]}}};\n\n')

        void_tags = []
        for i, (x, y, z) in enumerate(positions):
            tag = i + 2  # start after box ID 1
            void_tags.append(tag)
            f.write(f'Sphere({tag}) = {{{x}, {y}, {z}, {radius}}};\n')

        void_list = ', '.join(str(tag) for tag in void_tags)

        f.write('\n// --- Subtract voids ---\n')
        f.write(f'BooleanDifference{{ Volume{{1}}; Delete; }}{{ Volume{{{void_list}}}; Delete; }}\n\n')

        f.write('// --- Tag fluid volume ---\n')
        f.write('volumes() = Volume{:};\n')
        f.write('Physical Volume(\"Fluid\") = {volumes};\n\n')

        # No box surface tagging here (as requested)

        # Tag void surfaces together
        f.write('// --- Void surfaces ---\n')
        f.write('voidSurfaces[] = {};\n')
        for (x, y, z) in positions:
            rpad = radius + 1e-3
            f.write(
            f'voidSurfaces[] += Surface In BoundingBox{{{x - rpad}, {y - rpad}, {z - rpad}, {x + rpad}, {y + rpad}, {z + rpad}}};\n'
            )
        f.write('Physical Surface(1000) = {voidSurfaces[]};\n\n')

        f.write('Mesh.Algorithm = 6;\n')
        f.write(f'Mesh.MeshSizeMax = {h};\n')


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_voids(positions, radius, domain):
    def create_sphere(center, radius, resolution=5):
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, resolution),
                           np.linspace(0, np.pi, resolution))
        x = center[0] + radius * np.cos(u) * np.sin(v)
        y = center[1] + radius * np.sin(u) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        return x, y, z

    def draw_domain_box(ax, domain):
        Lx, Ly, Lz = domain
        # Define 8 corners of the box
        corners = np.array([
            [0, 0, 0], [Lx, 0, 0], [Lx, Ly, 0], [0, Ly, 0],
            [0, 0, Lz], [Lx, 0, Lz], [Lx, Ly, Lz], [0, Ly, Lz]
        ])
        # Define lines between corners (12 edges)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # top
            (0, 4), (1, 5), (2, 6), (3, 7)   # vertical
        ]
        for start, end in edges:
            ax.plot(*zip(corners[start], corners[end]), color='black', linewidth=1)

    # Plotting
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for pos in positions:
        x, y, z = create_sphere(pos, radius)
        ax.plot_surface(x, y, z, color='steelblue', edgecolor='k', linewidth=0.2, alpha=0.7)

    draw_domain_box(ax, domain)

    ax.set_xlim(0, domain[0])
    ax.set_ylim(0, domain[1])
    ax.set_zlim(0, domain[2])
    ax.set_box_aspect(domain)
    ax.set_title("BCC Lattice with Spherical Voids and Domain Wireframe")
    plt.tight_layout()
    plt.show()


import numpy as np

# def compute_bcc_positions(r, h, domain):
#     """
#     Compute BCC positions ensuring:
#     - The gap between spheres = i = a - 2r
#     - The domain edges are at distance i from nearest sphere surface
#     - i > 3h

#     Returns:
#         positions: array of (x, y, z)
#         a: spacing
#         i: gap between spheres and from spheres to walls
#         (nx, ny, nz): number of unit cells
#     """
#     Lx, Ly, Lz = domain
#     min_i = 3 * h

#     # Solve for max nx such that i = a - 2r > min_i
#     def get_max_n(L):
#         for n in range(1, 100):
#             a = L / (n + 1)
#             i = a - 2 * r
#             if i <= min_i:
#                 return n - 1  # previous was max valid
#         raise ValueError("Could not find suitable number of voids")

#     nx = get_max_n(Lx)
#     ny = get_max_n(Ly)
#     nz = get_max_n(Lz)

#     ax = Lx / (nx + 1)
#     ay = Ly / (ny + 1)
#     az = Lz / (nz + 1)

#     ix = ax - 2 * r
#     iy = ay - 2 * r
#     iz = az - 2 * r

#     # Use average a and i for isotropic spheres (or keep them separate if you want anisotropic spacing)
#     assert abs(ix - iy) < 1e-8 and abs(iy - iz) < 1e-8, "Anisotropic spacing detected"
#     a = ax
#     i = ix

#     centers = []
#     for ix_ in range(nx):
#         for iy_ in range(ny):
#             for iz_ in range(nz):
#                 x0 = r + i + ix_ * a
#                 y0 = r + i + iy_ * a
#                 z0 = r + i + iz_ * a
#                 centers.append((x0, y0, z0))  # corner

#                 # Body center
#                 x1 = x0 + a / 2
#                 y1 = y0 + a / 2
#                 z1 = z0 + a / 2
#                 if x1 + r + i <= Lx and y1 + r + i <= Ly and z1 + r + i <= Lz:
#                     centers.append((x1, y1, z1))

#     return np.array(centers), a, i, (nx, ny, nz)

import numpy as np

def compute_bcc_positions(r, h, domain):
    def find_max_n(L):
        for n in range(1, 1000):
            i = (L - 2 * r * n) / (n + 1)
            if i <= 3 * h:
                return n - 1
        raise ValueError("Domain too small or h too large")

    Lx, Ly, Lz = domain
    nx = find_max_n(Lx)
    ny = find_max_n(Ly)
    nz = find_max_n(Lz)

    ix = (Lx - 2 * r * nx) / (nx + 1)
    iy = (Ly - 2 * r * ny) / (ny + 1)
    iz = (Lz - 2 * r * nz) / (nz + 1)

    ax = 2 * r + ix
    ay = 2 * r + iy
    az = 2 * r + iz

    x_base = [ix + r + i * ax for i in range(nx)]
    y_base = [iy + r + j * ay for j in range(ny)]
    z_base = [iz + r + k * az for k in range(nz)]

    positions = []

    # Corner spheres
    for x in x_base:
        for y in y_base:
            for z in z_base:
                positions.append((x, y, z))

                # Body-centered sphere
                xc = x + ax / 2
                yc = y + ay / 2
                zc = z + az / 2
                if (xc + r + ix <= Lx) and (yc + r + iy <= Ly) and (zc + r + iz <= Lz):
                    positions.append((xc, yc, zc))

    return np.array(positions), (nx, ny, nz), (ax, ay, az), (ix, iy, iz)


# Example usage
r = 0.06
h = 0.01
domain = (1.0, 1.0, 1.0)

positions, a, i, (nx, ny, nz) = compute_bcc_positions(r, h, domain)
# plot_voids(positions, r, domain)

num_voids = len(positions)
volume_voids = num_voids * (4/3) * np.pi * r**3
packing_fraction = volume_voids / (domain[0] * domain[1] * domain[2])
packing_percentage = int(packing_fraction * 100)

print(f'Packing fraction: {packing_fraction:.4f}')

plot_voids(positions, r, domain)

write_gmsh(f'bcc-lattice-pf_{packing_percentage}.geo', positions, r, domain, h)
