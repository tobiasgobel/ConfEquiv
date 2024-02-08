from CGENN.algebra.cliffordalgebra import CliffordAlgebra
from objects import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def read_off(file):
    """
    Reads a .off file and returns vertices and faces as numpy arrays.
    """
    with open(file, 'r') as f:
        lines = f.readlines()
        if lines[0].strip() != 'OFF':
            raise ValueError('Not a valid OFF file')
        
        # Extract number of vertices, faces, and edges
        num_vertices, num_faces, _ = map(int, lines[1].strip().split())
        
        # Read vertices
        vertices = []
        for line in lines[2:2+num_vertices]:
            vertex = list(map(float, line.strip().split()))
            vertices.append(vertex)
        
        # Read faces
        faces = []
        for line in lines[2+num_vertices:]:
            face = list(map(int, line.strip().split()[1:]))
            faces.append(face)
        
        return torch.Tensor(vertices)/10, torch.Tensor(faces)

def visualize_shape(vertices, faces):
    """
    Visualize the shape defined by vertices and faces.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert vertices and faces to numpy arrays for plotting
    vertices_np = vertices.numpy()
    faces_np = np.array(faces.numpy(), dtype=int)

    # Plot vertices
    ax.scatter(vertices_np[:, 0], vertices_np[:, 1], vertices_np[:, 2], c='b', marker='.')

    # Plot faces
    # for face in faces_np:
    #     x = vertices_np[face, 0]
    #     y = vertices_np[face, 1]
    #     z = vertices_np[face, 2]
    #     ax.plot([x[0], x[1], x[2], x[0]], 
    #             [y[0], y[1], y[2], y[0]], 
    #             [z[0], z[1], z[2], z[0]], 'r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Shape Visualization')
    plt.show()


# N = 20
# d = 2
metric = [1,1,1]
# ca = CliffordAlgebra(metric)
# x = torch.randn(N, 2, d)
# x_cl = ca.embed_grade(x,1)
# print(x_cl.shape)

# Example usage
if __name__ == "__main__":
    file_path = 'Non-rigidSample/S2.off'
    vertices, faces = read_off(file_path)
    visualize_shape(vertices, faces)
    points = multivector_collection(vertices, metric)
    print(vertices)
    print(points.CGA_to_standard())



