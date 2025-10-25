import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# === Helper Functions ===
def read_nodes(filename):
    nodes = {}
    with open(filename, 'r') as f:
        for lineno, line in enumerate(f, start=1):
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) < 3:
                    print(f"Warning: Skipping malformed line {lineno} in {filename}: {line.strip()}")
                    continue
                try:
                    idx = int(parts[0])
                    x, y = float(parts[1]), float(parts[2])
                    nodes[idx] = (x, y)
                except ValueError as e:
                    print(f"Warning: Error parsing line {lineno} in {filename}: {e}")
    return nodes

def read_elements(filename):
    elements = []
    with open(filename, 'r') as f:
        for lineno, line in enumerate(f, start=1):
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) < 4:
                    print(f"Warning: Skipping malformed line {lineno} in {filename}: {line.strip()}")
                    continue
                try:
                    element = tuple(int(float(x)) for x in parts[1:4])  # skip element ID, take 3 node indices
                    elements.append(element)
                except ValueError as e:
                    print(f"Warning: Failed to parse line {lineno} in {filename}: {e}")
    return elements

def read_boundary_conditions(filename):
    bc = {}
    with open(filename, 'r') as f:
        for lineno, line in enumerate(f, start=1):
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) < 3:
                    print(f"Warning: Skipping malformed line {lineno} in {filename}: {line.strip()}")
                    continue
                try:
                    node = int(parts[0])
                    dof = 'x' if parts[1] == '1' else 'y'
                    val = float(parts[2])
                    bc[(node, dof)] = val
                except ValueError as e:
                    print(f"Warning: Error parsing line {lineno} in {filename}: {e}")
    return bc

def read_forces(filename):
    forces = {}
    with open(filename, 'r') as f:
        for lineno, line in enumerate(f, start=1):
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) < 3:
                    print(f"Warning: Skipping malformed line {lineno} in {filename}: {line.strip()}")
                    continue
                try:
                    node = int(parts[0])
                    dof = int(parts[1])
                    val = float(parts[2])
                    if dof not in [1, 2]:
                        print(f"Warning: Invalid DOF at line {lineno} in {filename}: {dof}")
                        continue
                    dof_label = 'x' if dof == 1 else 'y'
                    forces[(node, dof_label)] = val
                except ValueError as e:
                    print(f"Warning: Error parsing line {lineno} in {filename}: {e}")
    return forces

def build_global_stiffness_matrix(nodes, elements, E, nu):
    ndof = 2 * len(nodes)
    K = np.zeros((ndof, ndof))
    C = E / (1 - nu**2) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])
    for n1, n2, n3 in elements:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        x3, y3 = nodes[n3]
        A = 0.5 * ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
        B = np.array([
            [y2 - y3, 0, y3 - y1, 0, y1 - y2, 0],
            [0, x3 - x2, 0, x1 - x3, 0, x2 - x1],
            [x3 - x2, y2 - y3, x1 - x3, y3 - y1, x2 - x1, y1 - y2]
        ]) / (2 * A)
        ke = A * B.T @ C @ B
        edofs = [2*n1-2, 2*n1-1, 2*n2-2, 2*n2-1, 2*n3-2, 2*n3-1]
        for i in range(6):
            for j in range(6):
                K[edofs[i], edofs[j]] += ke[i, j]
    return K

def apply_boundary_conditions(K, F, bc):
    for (node, dof), val in bc.items():
        idx = 2 * (node - 1) + (0 if dof == 'x' else 1)
        K[idx, :] = 0
        K[:, idx] = 0
        K[idx, idx] = 1
        F[idx] = val
    return K, F

def compute_displacements(K, F):
    return np.linalg.solve(K, F)

def compute_strains_stresses(elements, nodes, displacements, E, nu):
    C = E / (1 - nu**2) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])
    strains, stresses = [], []
    for n1, n2, n3 in elements:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        x3, y3 = nodes[n3]
        A = 0.5 * ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
        B = np.array([
            [y2 - y3, 0, y3 - y1, 0, y1 - y2, 0],
            [0, x3 - x2, 0, x1 - x3, 0, x2 - x1],
            [x3 - x2, y2 - y3, x1 - x3, y3 - y1, x2 - x1, y1 - y2]
        ]) / (2 * A)
        u = np.array([
            displacements[2*n1-2], displacements[2*n1-1],
            displacements[2*n2-2], displacements[2*n2-1],
            displacements[2*n3-2], displacements[2*n3-1]
        ])
        strain = B @ u
        stress = C @ strain
        strains.append(strain)
        stresses.append(stress)
    return strains, stresses

def write_output(filename, nodes, displacements, elements, strains, stresses):
    with open(filename, 'w') as f:
        f.write("Nodal displacements\nnode# x y u v\n")
        for i, (x, y) in nodes.items():
            u = displacements[2*i-2]
            v = displacements[2*i-1]
            f.write(f"{i:4d} {x:10.6f} {y:10.6f} {u:10.6f} {v:10.6f}\n")
        f.write("\nElement strains and stresses\nele# exx eyy exy sigmaxx sigmayy sigmaxy\n")
        for idx, (strain, stress) in enumerate(zip(strains, stresses)):
            f.write(f"{idx+1:4d} {strain[0]:10.6f} {strain[1]:10.6f} {strain[2]:10.6f} {stress[0]:10.6f} {stress[1]:10.6f} {stress[2]:10.6f}\n")

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python tri_solver.py <nodes.txt> <elements.txt> <displacements.txt> <forces.txt>")
        sys.exit(1)

    nodes = read_nodes(sys.argv[1])
    elements = read_elements(sys.argv[2])
    bc = read_boundary_conditions(sys.argv[3])
    forces = read_forces(sys.argv[4])

    ndof = 2 * len(nodes)
    F = np.zeros(ndof)
    for (node, dof), val in forces.items():
        idx = 2 * (node - 1) + (0 if dof == 'x' else 1)
        F[idx] = val

    E = 1.0
    nu = 0.3

    K = build_global_stiffness_matrix(nodes, elements, E, nu)
    K, F = apply_boundary_conditions(K, F, bc)
    displacements = compute_displacements(K, F)
    strains, stresses = compute_strains_stresses(elements, nodes, displacements, E, nu)

    write_output("output.txt", nodes, displacements, elements, strains, stresses)

    print("Output written to output.txt")
