# adil
projects of adil
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm  # For color mapping
import tkinter as tk
from tkinter import messagebox, scrolledtext

# Finite Element Analysis Code (Function Definitions)

def D_matrix(E, nu):
    factor = E / ((1 + nu) * (1 - 2 * nu))
    D = factor * np.array([
        [1 - nu, nu, nu, 0, 0, 0],
        [nu, 1 - nu, nu, 0, 0, 0],
        [nu, nu, 1 - nu, 0, 0, 0],
        [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
        [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
        [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]
    ])
    return D

def shape_function_derivatives(xi, eta, zeta):
    dN_dxi = np.array([[-(1 - eta) * (1 - zeta),
                        (1 - eta) * (1 - zeta),
                        (1 + eta) * (1 - zeta),
                        -(1 + eta) * (1 - zeta),
                        -(1 - eta) * (1 + zeta),
                        (1 - eta) * (1 + zeta),
                        (1 + eta) * (1 + zeta),
                        -(1 + eta) * (1 + zeta)]]) * 0.125

    dN_deta = np.array([[-(1 - xi) * (1 - zeta),
                         -(1 + xi) * (1 - zeta),
                         (1 + xi) * (1 - zeta),
                         (1 - xi) * (1 - zeta),
                         -(1 - xi) * (1 + zeta),
                         -(1 + xi) * (1 + zeta),
                         (1 + xi) * (1 + zeta),
                         (1 - xi) * (1 + zeta)]]) * 0.125

    dN_dzeta = np.array([[-(1 - xi) * (1 - eta),
                          -(1 + xi) * (1 - eta),
                          -(1 + xi) * (1 + eta),
                          -(1 - xi) * (1 + eta),
                          (1 - xi) * (1 - eta),
                          (1 + xi) * (1 - eta),
                          (1 + xi) * (1 + eta),
                          (1 - xi) * (1 + eta)]]) * 0.125

    return dN_dxi, dN_deta, dN_dzeta

def element_stiffness(E, nu, nodes, element):
    D = D_matrix(E, nu)
    K_element = np.zeros((24, 24))
    gauss_points = [(-np.sqrt(1/3), -np.sqrt(1/3), -np.sqrt(1/3)),
                    (np.sqrt(1/3), -np.sqrt(1/3), -np.sqrt(1/3)),
                    (-np.sqrt(1/3), np.sqrt(1/3), -np.sqrt(1/3)),
                    (np.sqrt(1/3), np.sqrt(1/3), -np.sqrt(1/3)),
                    (-np.sqrt(1/3), -np.sqrt(1/3), np.sqrt(1/3)),
                    (np.sqrt(1/3), -np.sqrt(1/3), np.sqrt(1/3)),
                    (-np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3)),
                    (np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3))]

    for xi, eta, zeta in gauss_points:
        dN_dxi, dN_deta, dN_dzeta = shape_function_derivatives(xi, eta, zeta)
        J = np.zeros((3, 3))
        for i in range(8):
            J[0, 0] += dN_dxi[0, i] * nodes[element[i], 0]
            J[0, 1] += dN_dxi[0, i] * nodes[element[i], 1]
            J[0, 2] += dN_dxi[0, i] * nodes[element[i], 2]
            J[1, 0] += dN_deta[0, i] * nodes[element[i], 0]
            J[1, 1] += dN_deta[0, i] * nodes[element[i], 1]
            J[1, 2] += dN_deta[0, i] * nodes[element[i], 2]
            J[2, 0] += dN_dzeta[0, i] * nodes[element[i], 0]
            J[2, 1] += dN_dzeta[0, i] * nodes[element[i], 1]
            J[2, 2] += dN_dzeta[0, i] * nodes[element[i], 2]

        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        dN_dx = invJ @ np.vstack((dN_dxi, dN_deta, dN_dzeta))
        B = np.zeros((6, 24))
        for i in range(8):
            B[0, 3*i] = dN_dx[0, i]
            B[1, 3*i + 1] = dN_dx[1, i]
            B[2, 3*i + 2] = dN_dx[2, i]
            B[3, 3*i] = dN_dx[1, i]
            B[3, 3*i + 1] = dN_dx[0, i]
            B[4, 3*i + 1] = dN_dx[2, i]
            B[4, 3*i + 2] = dN_dx[1, i]
            B[5, 3*i] = dN_dx[2, i]
            B[5, 3*i + 2] = dN_dx[0, i]
        K_element += B.T @ D @ B * detJ
    return K_element

def assemble_global_stiffness(nodes, elements, E, nu):
    n_dof = 3 * len(nodes)
    K_global = np.zeros((n_dof, n_dof))
    for element in elements:
        K_elem = element_stiffness(E, nu, nodes, element)
        element_dofs = []
        for i in element:
            element_dofs.extend([3 * i, 3 * i + 1, 3 * i + 2])
        for i in range(24):
            for j in range(24):
                K_global[element_dofs[i], element_dofs[j]] += K_elem[i, j]
    return K_global

def apply_boundary_conditions(K_global, F_global, fixed_dofs):
    for dof in fixed_dofs:
        K_global[dof, :] = 0
        K_global[:, dof] = 0
        K_global[dof, dof] = 1
        F_global[dof] = 0

def calculate_displacements(length, num_elements, E, nu, load):
    element_size = length / num_elements
    nodes = []
    for i in range(num_elements + 1):
        x = i * element_size
        nodes.extend([[x, 0, 0], [x, 1/2, 0], [x, 1/2, 1/2], [x, 0, 1/2]])

    nodes = np.array(nodes)
    elements = []
    for i in range(num_elements):
        base_index = 4 * i
        elements.append([
            base_index, base_index + 1, base_index + 2, base_index + 3,
            base_index + 4, base_index + 5, base_index + 6, base_index + 7
        ])
    elements = np.array(elements)

    K_global = assemble_global_stiffness(nodes, elements, E, nu)

    F_global = np.zeros(3 * len(nodes))
    F_global[3 * (len(nodes) - 1) + 2] = load  # Apply load at the end of the bar
    F_global[3 * (len(nodes) - 1) + 1] = -load
    fixed_dofs = [0, 1, 2, 3, 4, 5, 6, 7]  # Fix nodes at the start

    apply_boundary_conditions(K_global, F_global, fixed_dofs)

    displacements = np.linalg.solve(K_global, F_global)

    return displacements, nodes, elements

def plot_deformed_bar(nodes, elements, displacements):
    # Calculate the deformed positions
    displacement_magnitudes = np.linalg.norm(displacements.reshape(-1, 3), axis=1)
    scale = 0.01 * np.max(nodes[:, 0]) / np.max(displacement_magnitudes)
    deformed_nodes = nodes + scale * displacements.reshape(-1, 3)

    # Function to plot a bar with a color map for deformation
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set up color map
    norm = plt.Normalize(np.min(displacement_magnitudes), np.max(displacement_magnitudes))
    cmap = cm.plasma

    # Plot deformed bar with color map based on displacement magnitude
    for element in elements:
        element_displacement_magnitude = np.mean(displacement_magnitudes[element])
        color = cmap(norm(element_displacement_magnitude))
        verts = [deformed_nodes[element]]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=color, linewidths=0.5, edgecolors='k', alpha=0.8))

    ax.set_title("Deformed Bar with Displacement Magnitude")
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(displacement_magnitudes)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Displacement Magnitude")

    plt.show()

# Tkinter Application

import tkinter as tk
from tkinter import messagebox, scrolledtext

class FEMApp:
    def __init__(self, master):
        self.master = master
        master.title("Finite Element Analysis")
        master.geometry("800x600")
        master.config(bg='black')  # Set the background color to black
        
        # Title Label
        self.title_label = tk.Label(master, text="Finite Element Analysis", font=("Arial", 24, "bold"), fg="white", bg='black')
        self.title_label.pack(pady=20)

        # Frame for input fields
        self.input_frame = tk.Frame(master, bg='black')
        self.input_frame.pack(pady=10)

        # Input fields for parameters
        self.length_label = tk.Label(self.input_frame, text="Bar Length:", fg="white", bg='black', font=("Arial", 12))
        self.length_label.grid(row=0, column=0, padx=10, pady=5)
        self.length_entry = tk.Entry(self.input_frame)
        self.length_entry.grid(row=0, column=1, padx=10, pady=5)

        self.num_elements_label = tk.Label(self.input_frame, text="Number of Elements:", fg="white", bg='black', font=("Arial", 12))
        self.num_elements_label.grid(row=1, column=0, padx=10, pady=5)
        self.num_elements_entry = tk.Entry(self.input_frame)
        self.num_elements_entry.grid(row=1, column=1, padx=10, pady=5)

        self.E_label = tk.Label(self.input_frame, text="Young's Modulus (E):", fg="white", bg='black', font=("Arial", 12))
        self.E_label.grid(row=2, column=0, padx=10, pady=5)
        self.E_entry = tk.Entry(self.input_frame)
        self.E_entry.grid(row=2, column=1, padx=10, pady=5)

        self.nu_label = tk.Label(self.input_frame, text="Poisson's Ratio (nu):", fg="white", bg='black', font=("Arial", 12))
        self.nu_label.grid(row=3, column=0, padx=10, pady=5)
        self.nu_entry = tk.Entry(self.input_frame)
        self.nu_entry.grid(row=3, column=1, padx=10, pady=5)

        self.load_label = tk.Label(self.input_frame, text="Load at the End:", fg="white", bg='black', font=("Arial", 12))
        self.load_label.grid(row=4, column=0, padx=10, pady=5)
        self.load_entry = tk.Entry(self.input_frame)
        self.load_entry.grid(row=4, column=1, padx=10, pady=5)

        # Calculate button
        self.calculate_button = tk.Button(master, text="Calculate Displacements", command=self.calculate, bg='blue', fg='white', font=("Arial", 12))
        self.calculate_button.pack(pady=20)

        # ScrolledText widget for results
        self.result_text = scrolledtext.ScrolledText(master, width=80, height=20, bg='black', fg='white', font=("Arial", 12), wrap=tk.WORD)
        self.result_text.pack(pady=10)
        self.result_text.config(state=tk.DISABLED)  # Start in a disabled state

    def calculate(self):
        try:
            # Get parameters from input fields
            length = float(self.length_entry.get())
            num_elements = int(self.num_elements_entry.get())
            E = float(self.E_entry.get())
            nu = float(self.nu_entry.get())
            load = float(self.load_entry.get())

            # Calculate displacements
            displacements, nodes, elements = calculate_displacements(length, num_elements, E, nu, load)

            # Clear previous results
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)  # Clear previous text

            # Show results in the text widget
            result_text = "Displacements at nodes:\n"
            result_text += f"{'Node':<10}{'u_x (m)':<15}{'u_y (m)':<15}{'u_z (m)':<15}\n"
            result_text += "-" * 60 + "\n"

            for i in range(len(nodes)):
                result_text += f"Node {i}: u_x = {displacements[3*i]:.6f}, u_y = {displacements[3*i+1]:.6f}, u_z = {displacements[3*i+2]:.6f}\n"
            
            self.result_text.insert(tk.END, result_text)  # Insert result text
            self.result_text.config(state=tk.DISABLED)  # Disable editing

            # Plot deformed bar
            plot_deformed_bar(nodes, elements, displacements)

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FEMApp(root)
    root.mainloop()
