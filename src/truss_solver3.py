import numpy as np
import pandas as pd

def solve_truss(nodes, elements, loads, constraints):
    """
    Solve a 2D truss problem with the specified properties.
    
    Parameters:
    nodes: Dictionary of node coordinates {node_id: {'x': x_coord, 'y': y_coord}}
    elements: List of dictionaries with element properties
    loads: Dictionary of applied loads {node_id: {'x': fx, 'y': fy}}
    constraints: Dictionary of constrained DOFs {node_id: {'x': True/False, 'y': True/False}}
    
    Returns:
    displacements, reactions, element_forces
    """
    # Number of nodes and DOFs
    num_nodes = len(nodes)
    ndof = num_nodes * 2  # 2 DOFs per node (x and y) for 2D truss
    
    # Initialize global stiffness matrix and force vector
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)
    
    # Global DOF mapping
    dof_map = {}
    for i, node_id in enumerate(sorted(nodes.keys())):
        dof_map[node_id] = {'x': 2*i, 'y': 2*i+1}
    
    # Assemble global stiffness matrix
    for elem in elements:
        n1 = elem['n1']
        n2 = elem['n2']
        
        # Element geometry
        x1, y1 = nodes[n1]['x'], nodes[n1]['y']
        x2, y2 = nodes[n2]['x'], nodes[n2]['y']
        
        dx = x2 - x1
        dy = y2 - y1
        L = np.sqrt(dx**2 + dy**2)
        
        # Direction cosines
        c = dx / L
        s = dy / L
        
        # Element properties
        E = elem['E']
        A = elem['A']
        EA_L = E * A / L
        
        # Element stiffness matrix in global coordinates
        k_local = np.array([
            [c*c, c*s, -c*c, -c*s],
            [c*s, s*s, -c*s, -s*s],
            [-c*c, -c*s, c*c, c*s],
            [-c*s, -s*s, c*s, s*s]
        ]) * EA_L
        
        # Get DOF indices
        dofs = [
            dof_map[n1]['x'], dof_map[n1]['y'],
            dof_map[n2]['x'], dof_map[n2]['y']
        ]
        
        # Assemble into global stiffness matrix
        for i in range(4):
            for j in range(4):
                K[dofs[i], dofs[j]] += k_local[i, j]
    
    # Apply loads
    for node_id, load in loads.items():
        if 'x' in load:
            F[dof_map[node_id]['x']] = load['x']
        if 'y' in load:
            F[dof_map[node_id]['y']] = load['y']
    
    # Apply constraints - partition the stiffness matrix
    free_dofs = []
    fixed_dofs = []
    
    for node_id in sorted(nodes.keys()):
        if node_id in constraints:
            if constraints[node_id].get('x', False):
                fixed_dofs.append(dof_map[node_id]['x'])
            else:
                free_dofs.append(dof_map[node_id]['x'])
                
            if constraints[node_id].get('y', False):
                fixed_dofs.append(dof_map[node_id]['y'])
            else:
                free_dofs.append(dof_map[node_id]['y'])
        else:
            free_dofs.append(dof_map[node_id]['x'])
            free_dofs.append(dof_map[node_id]['y'])
    
    # Solve for displacements
    if len(free_dofs) > 0:
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]
        
        # Solve the system
        u_f = np.linalg.solve(K_ff, F_f)
        
        # Full displacement vector
        u = np.zeros(ndof)
        for i, dof in enumerate(free_dofs):
            u[dof] = u_f[i]
    else:
        u = np.zeros(ndof)
    
    # Calculate reactions
    reactions = K @ u - F
    
    # Format displacement results
    displacements = {}
    for node_id in sorted(nodes.keys()):
        displacements[node_id] = {
            'x': u[dof_map[node_id]['x']],
            'y': u[dof_map[node_id]['y']]
        }
    
    # Format reaction results
    reaction_forces = {}
    for node_id in sorted(nodes.keys()):
        if node_id in constraints:
            reaction_forces[node_id] = {
                'x': reactions[dof_map[node_id]['x']] if constraints[node_id].get('x', False) else 0,
                'y': reactions[dof_map[node_id]['y']] if constraints[node_id].get('y', False) else 0
            }
    
    # Calculate element forces
    element_forces = []
    for elem in elements:
        n1 = elem['n1']
        n2 = elem['n2']
        
        # Element geometry
        x1, y1 = nodes[n1]['x'], nodes[n1]['y']
        x2, y2 = nodes[n2]['x'], nodes[n2]['y']
        
        dx = x2 - x1
        dy = y2 - y1
        L = np.sqrt(dx**2 + dy**2)
        
        # Direction cosines
        c = dx / L
        s = dy / L
        
        # Get displacements for this element
        u1x = displacements[n1]['x']
        u1y = displacements[n1]['y']
        u2x = displacements[n2]['x']
        u2y = displacements[n2]['y']
        
        # Calculate axial force
        du = c * (u2x - u1x) + s * (u2y - u1y)
        axial_strain = du / L
        axial_force = axial_strain * elem['E'] * elem['A']
        
        element_forces.append({
            'id': elem.get('id', None),
            'n1': n1,
            'n2': n2,
            'strain': axial_strain,
            'force': axial_force,
            'stress': axial_force / elem['A']
        })
    
    return displacements, reaction_forces, element_forces

def setup_hw5_truss_problem(L, E, A, P):
    """
    Setup the homework 5 problem for a truss structure (EI = 0).
    
    Parameters:
    L: Length unit
    E: Young's modulus
    A: Cross-sectional area
    P: Applied load
    
    Returns:
    nodes, elements, loads, constraints
    """
    # Node coordinates
    nodes = {
        1: {'x': 0, 'y': 0},        # Bottom left support
        2: {'x': 0, 'y': 3*L},      # Top node
        3: {'x': 2*L, 'y': 0},      # Load application point
        4: {'x': 4*L, 'y': 0}       # Bottom right support
    }
    
    # Elements
    elements = [
        {'id': 1, 'n1': 1, 'n2': 2, 'E': E, 'A': A},  # Left vertical member
        {'id': 2, 'n1': 2, 'n2': 3, 'E': E, 'A': A},  # Diagonal member
        {'id': 3, 'n1': 3, 'n2': 4, 'E': E, 'A': A},  # Right horizontal member
        {'id': 4, 'n1': 1, 'n2': 3, 'E': E, 'A': A}   # Bottom horizontal member
    ]
    
    # Loads
    loads = {
        3: {'x': 0, 'y': -P}  # Vertical downward force at node 3
    }
    
    # Constraints
    constraints = {
        1: {'x': True, 'y': True},  # Fixed support at node 1
        4: {'x': True, 'y': True}   # Roller support at node 4 (for truss, both x and y are fixed)
    }
    
    return nodes, elements, loads, constraints

def run_truss_analysis(L, E, A, P):
    """
    Run analysis for the truss structure (EI = 0).
    
    Parameters:
    L: Length unit
    E: Young's modulus
    A: Cross-sectional area
    P: Applied load
    
    Returns:
    Normalized displacements and element forces
    """
    # Setup the problem
    nodes, elements, loads, constraints = setup_hw5_truss_problem(L, E, A, P)
    
    # Solve the truss
    displacements, reactions, element_forces = solve_truss(nodes, elements, loads, constraints)
    
    # Normalization factor
    norm_factor = P * L / (E * A)
    
    # Extract and normalize nodal displacements
    normalized_results = {}
    for node_id, disp in displacements.items():
        normalized_results[f"u{node_id}/(norm)"] = disp['x'] / norm_factor
        normalized_results[f"v{node_id}/(norm)"] = disp['y'] / norm_factor
        # No rotation for truss elements
        normalized_results[f"θ{node_id}/(norm)"] = 0.0
    
    # Extract axial force in vertical element
    vertical_element = next(ef for ef in element_forces if ef['id'] == 1)
    vertical_axial_force = vertical_element['force']
    
    return normalized_results, vertical_axial_force

if __name__ == "__main__":
    # Example parameters
    L = 1.0      # Length unit
    E = 200e9    # Young's modulus (Pa)
    A = 0.01     # Cross-sectional area (m²)
    P = 1000.0   # Applied load (N)
    
    print("Running truss analysis (EI = 0)...")
    normalized_displacements, vertical_force = run_truss_analysis(L, E, A, P)
    
    # Display results
    print("\nNormalized Displacements (Truss, EI = 0):")
    for dof, value in sorted(normalized_displacements.items()):
        print(f"{dof}: {value:.6e}")
    
    print(f"\nAxial Force in Vertical Element (Truss): {vertical_force:.6f}")
    
    # Save results to CSV
    df = pd.DataFrame([normalized_displacements])
    df = df.T.reset_index()
    df.columns = ['DOF', 'EI = 0 (truss)']
    print("\nResults saved to 'truss_results.csv'")
    df.to_csv("truss_results.csv", index=False)