#!/usr/bin/env python3
"""
Simple example demonstrating the FEM solver capabilities.
This script runs a basic truss analysis problem.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from truss_solver3 import solve_truss, setup_hw5_truss_problem

def main():
    print("Finite Element Method Solver - Example")
    print("=" * 40)
    
    # Setup a simple truss problem
    L = 1.0      # Length unit (m)
    E = 200e9    # Young's modulus (Pa)
    A = 0.01     # Cross-sectional area (m²)
    P = 1000.0   # Applied load (N)
    
    print(f"Problem parameters:")
    print(f"  Length: {L} m")
    print(f"  Young's modulus: {E/1e9:.0f} GPa")
    print(f"  Cross-sectional area: {A*10000:.1f} cm²")
    print(f"  Applied load: {P} N")
    print()
    
    # Setup the problem
    nodes, elements, loads, constraints = setup_hw5_truss_problem(L, E, A, P)
    
    print("Node coordinates:")
    for node_id, coords in nodes.items():
        print(f"  Node {node_id}: ({coords['x']:.1f}, {coords['y']:.1f})")
    print()
    
    print("Elements:")
    for elem in elements:
        print(f"  Element {elem['id']}: Nodes {elem['n1']}-{elem['n2']}")
    print()
    
    # Solve the truss
    print("Solving...")
    displacements, reactions, element_forces = solve_truss(nodes, elements, loads, constraints)
    
    print("Results:")
    print("-" * 20)
    print("Nodal Displacements:")
    for node_id, disp in displacements.items():
        print(f"  Node {node_id}: u = {disp['x']:.6f} m, v = {disp['y']:.6f} m")
    
    print("\nElement Forces:")
    for elem in element_forces:
        print(f"  Element {elem['id']}: Force = {elem['force']:.2f} N, Stress = {elem['stress']/1e6:.2f} MPa")
    
    print("\nReaction Forces:")
    for node_id, reaction in reactions.items():
        if reaction['x'] != 0 or reaction['y'] != 0:
            print(f"  Node {node_id}: Fx = {reaction['x']:.2f} N, Fy = {reaction['y']:.2f} N")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
