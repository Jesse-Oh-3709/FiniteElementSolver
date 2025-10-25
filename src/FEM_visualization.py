import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from scipy.interpolate import interp1d

def read_nodes(filepath):
    """Read node coordinates from file"""
    nodes = {}
    with open(filepath, 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if parts:
                node = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                nodes[node] = (x, y)
    return nodes

def read_elements(filepath):
    """Read element connectivity and material properties"""
    elements = []
    with open(filepath, 'r') as f:
        header = f.readline().strip().split()
        num_elements = int(header[0])
        E = float(header[1])
        nu = float(header[2])
        for line in f:
            parts = line.strip().split()
            if parts:
                ele = int(parts[0])
                n1 = int(parts[1])
                n2 = int(parts[2])
                n3 = int(parts[3])
                elements.append((ele, n1, n2, n3, E, nu))
    return elements

def read_displacements(filepath):
    """Read nodal displacements from solution file"""
    displacements = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        section = None
        
        for line in lines:
            line = line.strip()
            
            if "Nodal displacements" in line:
                section = "displacements"
                continue
            elif "Element stresses" in line:
                section = "stresses"
                continue
            elif not line or line.startswith("node#") or line.startswith("ele#"):
                continue
            
            if section == "displacements" and len(line.split()) >= 5:
                parts = line.split()
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                u = float(parts[3])
                v = float(parts[4])
                displacements[node_id] = (x, y, u, v)
    
    return displacements

def read_stresses(filepath):
    """Read element stresses from solution file"""
    stresses = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        section = None
        
        for line in lines:
            line = line.strip()
            
            if "Nodal displacements" in line:
                section = "displacements"
                continue
            elif "Element stresses" in line:
                section = "stresses"
                continue
            elif not line or line.startswith("node#") or line.startswith("ele#"):
                continue
            
            if section == "stresses" and len(line.split()) >= 4:
                parts = line.split()
                elem_id = int(parts[0])
                sigmaxx = float(parts[1])
                sigmayy = float(parts[2])
                sigmaxy = float(parts[3])
                stresses[elem_id] = (sigmaxx, sigmayy, sigmaxy)
    
    return stresses

def plot_mesh(nodes, elements, title="Finite Element Mesh"):
    """Plot the finite element mesh"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create patches for elements
    patches = []
    for ele_id, n1, n2, n3, _, _ in elements:
        vertices = np.array([nodes[n1], nodes[n2], nodes[n3]])
        polygon = Polygon(vertices, True)
        patches.append(polygon)
    
    # Add elements to plot
    p = PatchCollection(patches, alpha=0.4, edgecolor='black', facecolor='none')
    ax.add_collection(p)
    
    # Plot nodes
    for node_id, (x, y) in nodes.items():
        ax.plot(x, y, 'o', color='blue', markersize=8)
        ax.text(x+0.02, y+0.02, str(node_id), fontsize=12)
    
    # Set axis limits with some padding
    coords = np.array(list(nodes.values()))
    xmin, ymin = coords.min(axis=0) - 0.1
    xmax, ymax = coords.max(axis=0) + 0.1
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Add labels and title
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal')
    
    plt.grid(True)
    plt.savefig('mesh.png', dpi=300)
    plt.close()

def plot_deformed_shape(nodes, elements, displacements, scale_factor=0.5, title=None):
    """Plot the original and deformed mesh"""
    if title is None:
        title = f"Deformed Shape (Scale Factor = {scale_factor})"
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create patches for original elements
    orig_patches = []
    for ele_id, n1, n2, n3, _, _ in elements:
        vertices = np.array([nodes[n1], nodes[n2], nodes[n3]])
        polygon = Polygon(vertices, True)
        orig_patches.append(polygon)
    
    # Add original elements to plot
    p_orig = PatchCollection(orig_patches, alpha=0.3, edgecolor='gray', facecolor='none')
    ax.add_collection(p_orig)
    
    # Create patches for deformed elements
    def_patches = []
    for ele_id, n1, n2, n3, _, _ in elements:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        x3, y3 = nodes[n3]
        
        u1, v1 = displacements[n1][2], displacements[n1][3]
        u2, v2 = displacements[n2][2], displacements[n2][3]
        u3, v3 = displacements[n3][2], displacements[n3][3]
        
        # Scale displacements
        u1 *= scale_factor
        v1 *= scale_factor
        u2 *= scale_factor
        v2 *= scale_factor
        u3 *= scale_factor
        v3 *= scale_factor
        
        # Deformed coordinates
        x1_def, y1_def = x1 + u1, y1 + v1
        x2_def, y2_def = x2 + u2, y2 + v2
        x3_def, y3_def = x3 + u3, y3 + v3
        
        vertices = np.array([(x1_def, y1_def), (x2_def, y2_def), (x3_def, y3_def)])
        polygon = Polygon(vertices, True)
        def_patches.append(polygon)
    
    # Add deformed elements to plot
    p_def = PatchCollection(def_patches, alpha=0.7, edgecolor='blue', facecolor='none')
    ax.add_collection(p_def)
    
    # Plot original nodes
    for node_id, (x, y) in nodes.items():
        ax.plot(x, y, 'o', color='gray', markersize=5, alpha=0.5)
    
    # Plot deformed nodes
    for node_id, (x, y, u, v) in displacements.items():
        x_def = x + u * scale_factor
        y_def = y + v * scale_factor
        ax.plot(x_def, y_def, 'o', color='blue', markersize=7)
        ax.text(x_def+0.02, y_def+0.02, str(node_id), fontsize=12)
    
    # Set axis limits with some padding
    coords = np.array(list(nodes.values()))
    deformed_coords = np.array([(disp[0] + disp[2]*scale_factor, disp[1] + disp[3]*scale_factor) 
                                for disp in displacements.values()])
    
    all_coords = np.vstack([coords, deformed_coords])
    xmin, ymin = all_coords.min(axis=0) - 0.2
    xmax, ymax = all_coords.max(axis=0) + 0.2
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Add labels and title
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', marker='o', linestyle='None', 
               markersize=6, label='Original Nodes'),
        Line2D([0], [0], color='blue', marker='o', linestyle='None', 
               markersize=8, label='Deformed Nodes'),
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2,
               label='Original Mesh'),
        Line2D([0], [0], color='blue', linestyle='-', linewidth=2,
               label='Deformed Mesh')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.grid(True)
    plt.savefig('deformed_shape.png', dpi=300)
    plt.close()

def plot_stress_contour(nodes, elements, displacements, stresses, stress_component='xx', 
                        scale_factor=0.5, title=None):
    """Plot stress contours on the deformed mesh"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Extract stress values based on component
    if stress_component == 'xx':
        idx = 0
        component_name = r"$\sigma_{xx}$"
    elif stress_component == 'yy':
        idx = 1
        component_name = r"$\sigma_{yy}$"
    elif stress_component == 'xy':
        idx = 2
        component_name = r"$\tau_{xy}$"
    else:
        raise ValueError("Invalid stress component")
    
    # Create deformed patches for elements
    def_patches = []
    stress_values = []
    
    for ele_id, n1, n2, n3, _, _ in elements:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        x3, y3 = nodes[n3]
        
        u1, v1 = displacements[n1][2], displacements[n1][3]
        u2, v2 = displacements[n2][2], displacements[n2][3]
        u3, v3 = displacements[n3][2], displacements[n3][3]
        
        # Scale displacements
        u1 *= scale_factor
        v1 *= scale_factor
        u2 *= scale_factor
        v2 *= scale_factor
        u3 *= scale_factor
        v3 *= scale_factor
        
        # Deformed coordinates
        x1_def, y1_def = x1 + u1, y1 + v1
        x2_def, y2_def = x2 + u2, y2 + v2
        x3_def, y3_def = x3 + u3, y3 + v3
        
        vertices = np.array([(x1_def, y1_def), (x2_def, y2_def), (x3_def, y3_def)])
        polygon = Polygon(vertices, True)
        def_patches.append(polygon)
        
        # Get stress value for this element
        stress_values.append(stresses[ele_id][idx])
    
    # Create color map for stress values
    cmap = cm.get_cmap('coolwarm')
    p_def = PatchCollection(def_patches, cmap=cmap, alpha=0.8, edgecolor='black')
    p_def.set_array(np.array(stress_values))
    ax.add_collection(p_def)
    
    # Add color bar
    cbar = plt.colorbar(p_def, ax=ax)
    cbar.set_label(f'{component_name} Stress', fontsize=14)
    
    # Plot deformed nodes
    for node_id, (x, y, u, v) in displacements.items():
        x_def = x + u * scale_factor
        y_def = y + v * scale_factor
        ax.plot(x_def, y_def, 'o', color='black', markersize=6)
        ax.text(x_def+0.02, y_def+0.02, str(node_id), fontsize=12)
    
    # Set axis limits with some padding
    deformed_coords = np.array([(disp[0] + disp[2]*scale_factor, disp[1] + disp[3]*scale_factor) 
                                for disp in displacements.values()])
    
    xmin, ymin = deformed_coords.min(axis=0) - 0.1
    xmax, ymax = deformed_coords.max(axis=0) + 0.1
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Add labels and title
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    
    if title is None:
        title = f"{component_name} Stress Contour (scale factor = {scale_factor})"
    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal')
    
    plt.grid(True)
    plt.savefig(f'stress_{stress_component}.png', dpi=300)
    plt.close()

def plot_stress_along_axis(nodes, elements, stresses, axis='x', title=None):
    """Plot stress variation along an axis using normalized coordinates (x/R or y/R)"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Find the maximum coordinate value (R)
    coords = np.array(list(nodes.values()))
    if axis == 'x':
        R = np.max(coords[:, 0])
    else:
        R = np.max(coords[:, 1])
    
    # Calculate element centroids and their normalized coordinates
    centroids = []
    for ele_id, n1, n2, n3, _, _ in elements:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        x3, y3 = nodes[n3]
        
        centroid_x = (x1 + x2 + x3) / 3
        centroid_y = (y1 + y2 + y3) / 3
        
        if axis == 'x':
            normalized_coord = centroid_x / R
            centroids.append((ele_id, normalized_coord, centroid_y))
        else:
            normalized_coord = centroid_y / R
            centroids.append((ele_id, centroid_x, normalized_coord))
    
    # Sort centroids by the normalized coordinate
    centroids.sort(key=lambda c: c[1] if axis == 'x' else c[2])
    
    # Extract stress data and normalized coordinates
    normalized_coords = []
    sigma_xx_values = []
    sigma_yy_values = []
    sigma_xy_values = []
    
    for centroid in centroids:
        ele_id = centroid[0]
        if axis == 'x':
            normalized_coords.append(centroid[1])  # x/R
        else:
            normalized_coords.append(centroid[2])  # y/R
        
        sigma_xx, sigma_yy, sigma_xy = stresses[ele_id]
        sigma_xx_values.append(sigma_xx)
        sigma_yy_values.append(sigma_yy)
        sigma_xy_values.append(sigma_xy)
    
    # Plot stress components vs normalized coordinate
    ax.plot(normalized_coords, sigma_xx_values, 'ro-', linewidth=2, label=r'$\sigma_{xx}$')
    ax.plot(normalized_coords, sigma_yy_values, 'bo-', linewidth=2, label=r'$\sigma_{yy}$')
    ax.plot(normalized_coords, sigma_xy_values, 'go-', linewidth=2, label=r'$\tau_{xy}$')
    
    # Add labels and title
    if axis == 'x':
        ax.set_xlabel(r'$x/R$', fontsize=14)
    else:
        ax.set_xlabel(r'$y/R$', fontsize=14)
    ax.set_ylabel('Stress', fontsize=14)
    
    if title is None:
        title = f"Stress Variation Along {axis.upper()}-Axis"
    ax.set_title(title, fontsize=16)
    
    ax.legend(fontsize=12)
    ax.grid(True)
    
    # Add vertical lines at key positions
    ax.axvline(x=1, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=3, color='k', linestyle='--', alpha=0.5)
    
    plt.savefig(f'stress_along_{axis}.png', dpi=300)
    plt.close()

def extrapolate_stresses(nodes, elements, stresses, location='edge'):
    """Extrapolate stresses to edge points (x/R = 1 or y/R = 1)"""
    # Find the maximum coordinate value (R)
    coords = np.array(list(nodes.values()))
    R_x = np.max(coords[:, 0])
    R_y = np.max(coords[:, 1])
    
    # Calculate element centroids and their normalized coordinates
    x_centroids = []
    y_centroids = []
    stress_data = []
    
    for ele_id, n1, n2, n3, _, _ in elements:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        x3, y3 = nodes[n3]
        
        centroid_x = (x1 + x2 + x3) / 3
        centroid_y = (y1 + y2 + y3) / 3
        
        x_norm = centroid_x / R_x
        y_norm = centroid_y / R_y
        
        sigmaxx, sigmayy, sigmaxy = stresses[ele_id]
        stress_data.append((ele_id, x_norm, y_norm, sigmaxx, sigmayy, sigmaxy))
        
        if y_norm < 0.1:  # Points close to x-axis
            x_centroids.append((x_norm, sigmaxx, sigmayy, sigmaxy))
        
        if x_norm < 0.1:  # Points close to y-axis
            y_centroids.append((y_norm, sigmaxx, sigmayy, sigmaxy))
    
    # Sort by normalized coordinate
    x_centroids.sort(key=lambda c: c[0])
    y_centroids.sort(key=lambda c: c[0])
    
    # Linear extrapolation to x/R = 1 (if we have at least 2 points)
    extrapolated_x = None
    if len(x_centroids) >= 2:
        # Get the two closest points to x/R = 1
        x1, sx1_xx, sx1_yy, sx1_xy = x_centroids[-2]
        x2, sx2_xx, sx2_yy, sx2_xy = x_centroids[-1]
        
        # Linear extrapolation formula: y - y1 = ((y2 - y1) / (x2 - x1)) * (x - x1)
        # For x/R = 1
        ex_xx = sx1_xx + ((sx2_xx - sx1_xx) / (x2 - x1)) * (1 - x1)
        ex_yy = sx1_yy + ((sx2_yy - sx1_yy) / (x2 - x1)) * (1 - x1)
        ex_xy = sx1_xy + ((sx2_xy - sx1_xy) / (x2 - x1)) * (1 - x1)
        
        extrapolated_x = (ex_xx, ex_yy, ex_xy)
    
    # Linear extrapolation to y/R = 1 (if we have at least 2 points)
    extrapolated_y = None
    if len(y_centroids) >= 2:
        # Get the two closest points to y/R = 1
        y1, sy1_xx, sy1_yy, sy1_xy = y_centroids[-2]
        y2, sy2_xx, sy2_yy, sy2_xy = y_centroids[-1]
        
        # Linear extrapolation formula
        ey_xx = sy1_xx + ((sy2_xx - sy1_xx) / (y2 - y1)) * (1 - y1)
        ey_yy = sy1_yy + ((sy2_yy - sy1_yy) / (y2 - y1)) * (1 - y1)
        ey_xy = sy1_xy + ((sy2_xy - sy1_xy) / (y2 - y1)) * (1 - y1)
        
        extrapolated_y = (ey_xx, ey_yy, ey_xy)
    
    return extrapolated_x, extrapolated_y

def main():
    # Read input files
    nodes = read_nodes('nodes.txt')
    elements = read_elements('elements.txt')
    
    # Read solution files
    displacements = read_displacements('output.txt')
    stresses = read_stresses('output.txt')
    
    # Plot mesh
    plot_mesh(nodes, elements, title="Triangular Finite Element Mesh")
    
    # Plot deformed shape with scaling factor of 0.5
    scaling_factor = 0.5
    plot_deformed_shape(nodes, elements, displacements, scale_factor=scaling_factor, 
                       title=f"Deformed Shape (Scale Factor = {scaling_factor})")
    
    # Plot stress contours
    plot_stress_contour(nodes, elements, displacements, stresses, 
                       stress_component='xx', scale_factor=scaling_factor)
    plot_stress_contour(nodes, elements, displacements, stresses, 
                       stress_component='yy', scale_factor=scaling_factor)
    plot_stress_contour(nodes, elements, displacements, stresses, 
                       stress_component='xy', scale_factor=scaling_factor)
    
    # Plot stress along normalized axes (x/R and y/R)
    plot_stress_along_axis(nodes, elements, stresses, axis='x')
    plot_stress_along_axis(nodes, elements, stresses, axis='y')
    
    # Extrapolate stresses to edge points
    extrapolated_x, extrapolated_y = extrapolate_stresses(nodes, elements, stresses)
    
    print("Extrapolated stresses at x/R = 1:")
    if extrapolated_x:
        print(f"σ_xx = {extrapolated_x[0]:.6f}, σ_yy = {extrapolated_x[1]:.6f}, τ_xy = {extrapolated_x[2]:.6f}")
    else:
        print("Insufficient data for extrapolation.")
    
    print("\nExtrapolated stresses at y/R = 1:")
    if extrapolated_y:
        print(f"σ_xx = {extrapolated_y[0]:.6f}, σ_yy = {extrapolated_y[1]:.6f}, τ_xy = {extrapolated_y[2]:.6f}")
    else:
        print("Insufficient data for extrapolation.")

if __name__ == '__main__':
    main()