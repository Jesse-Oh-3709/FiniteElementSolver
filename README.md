# Finite Element Method (FEM) Solver

A comprehensive Finite Element Analysis solver implemented in Python and MATLAB for structural analysis of 2D and 3D problems.

## Features

- **Multiple Element Types**: Support for truss, frame, and triangular elements
- **Matrix Assembly**: Efficient global stiffness matrix assembly
- **Boundary Conditions**: Flexible application of displacement and force boundary conditions
- **Post-processing**: Displacement, stress, and strain calculation
- **Visualization**: Built-in plotting and visualization tools
- **Cross-platform**: Python and MATLAB implementations

## Solver Types

### 1. Truss Solver (`truss_solver3.py`)
- 2D truss elements with axial forces only
- Pin-jointed connections
- Ideal for bridge and roof structures

### 2. Frame Solver (`frame_solver.py`)
- 2D frame elements with axial, shear, and bending
- Rigid connections
- Suitable for building frames and beams

### 3. Triangular Solver (`tri_solver.py`)
- 2D triangular elements for plane stress/strain
- Linear triangular elements
- Ideal for complex geometries and stress analysis

## Installation

### Python Dependencies
```bash
pip install numpy matplotlib scipy pandas
```

### MATLAB Requirements
- MATLAB R2018b or later
- Statistics and Machine Learning Toolbox (for some advanced features)

## Usage

### Python Examples

#### Truss Analysis
```python
from src.truss_solver3 import solve_truss, setup_hw5_truss_problem

# Setup problem
nodes, elements, loads, constraints = setup_hw5_truss_problem(L=1.0, E=200e9, A=0.01, P=1000.0)

# Solve
displacements, reactions, element_forces = solve_truss(nodes, elements, loads, constraints)
```

#### Frame Analysis
```python
from src.frame_solver import solve_fem_frame

# Solve frame problem
solve_fem_frame("examples/nodes.txt", "examples/elements.txt", 
                "examples/forces.txt", "examples/displacements.txt", "output.txt")
```

#### Triangular Element Analysis
```python
from src.tri_solver import main
import sys

# Run triangular solver
sys.argv = ['tri_solver.py', 'examples/nodes.txt', 'examples/elements.txt', 
            'examples/displacements.txt', 'examples/forces.txt']
main()
```

### MATLAB Examples

```matlab
% Run main MATLAB solver
main()

% Or run specific homework problem
ASE330m_HW3()
```

## Input File Format

### Nodes File (`nodes.txt`)
```
# node_id x_coordinate y_coordinate
1 0.0 0.0
2 1.0 0.0
3 0.5 1.0
```

### Elements File (`elements.txt`)
```
# element_id node1 node2 [node3] E A [I]
1 1 2 200e9 0.01
2 2 3 200e9 0.01
```

### Forces File (`forces.txt`)
```
# node_id dof value
3 2 -1000.0
```

### Displacements File (`displacements.txt`)
```
# node_id dof value
1 1 0.0
1 2 0.0
```

## Visualization

The `FEM_visualization.py` module provides comprehensive plotting capabilities:

```python
from src.FEM_visualization import plot_mesh, plot_deformed_shape, plot_stress_contour

# Plot original mesh
plot_mesh(nodes, elements)

# Plot deformed shape
plot_deformed_shape(nodes, elements, displacements, scale_factor=0.5)

# Plot stress contours
plot_stress_contour(nodes, elements, displacements, stresses, stress_component='xx')
```

## Examples

The `examples/` directory contains sample problems:
- Simple truss structures
- Frame analysis problems
- Triangular element meshes
- Various boundary condition setups

## Documentation

The `docs/` directory contains:
- Course lecture notes and derivations
- Homework problem statements
- Theoretical background materials

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Author

**Jesse Oh**  
Computational Engineering  
The University of Texas at Austin

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Future Work

- [ ] 3D element support
- [ ] Nonlinear analysis capabilities
- [ ] Parallel processing for large problems
- [ ] GUI interface
- [ ] Integration with CAD software
- [ ] Advanced post-processing features

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Based on finite element theory from COE 321K course at UT Austin
- Inspired by classical structural analysis methods
- Built for educational and research purposes
