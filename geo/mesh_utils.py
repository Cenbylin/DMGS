import torch

@torch.no_grad()
def generate_barycentric(level):
    points = []
    n = level + 2
    # Iterate over the divisions, skipping the edges
    for i in range(1, n):
        for j in range(1, n - i):
            # Barycentric coordinates based on the current iteration
            barycentric_coord = torch.tensor([i, j, n - i - j]) / n
            points.append(barycentric_coord)
    return torch.stack(points, dim=0)

@torch.no_grad()
def generate_barycentric_v2(num, device='cpu'):
    if num == 1:
        barycentric_coords = torch.tensor([
            [1/3, 1/3, 1/3],
        ], device=device)
        radius = 1. / (2.*(3**0.5))
    elif num == 3:
        # barycentric_coords = torch.tensor([
        #     [1/2, 1/4, 1/4], [1/4, 1/2, 1/4], [1/4, 1/4, 1/2]], 
        #      device=device)  from SuGaR
        barycentric_coords = torch.tensor([
            [(3-(3**0.5))/6, (3-(3**0.5))/6, (3**0.5)/3], 
            [(3-(3**0.5))/6, (3**0.5)/3, (3-(3**0.5))/6], 
            [(3**0.5)/3, (3-(3**0.5))/6, (3-(3**0.5))/6]
        ], device=device)
        radius = 1. / (2. + 2.*(3**0.5))
    elif num == 6:
        barycentric_coords = torch.tensor(
            [[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3],
             [1/6, 5/12, 5/12], [5/12, 1/6, 5/12], [5/12, 5/12, 1/6]], 
             device=device)
        radius = 1 / (4. + 2.*(3**0.5))
    else:
        raise NotImplementedError
    return barycentric_coords, radius


def face_normals(vertices, faces, unit=False):
    # args: vertices[N_v, 3], faces[N_f, 3]
    face_vertices = vertices[faces]
    if face_vertices.shape[-2] != 3:
        raise NotImplementedError("face_normals is only implemented for triangle meshes")
    edges_dist0 = face_vertices[:, 1, :] - face_vertices[:, 0, :]
    edges_dist1 = face_vertices[:, 2, :] - face_vertices[:, 0, :]
    face_normals = torch.cross(edges_dist0, edges_dist1, dim=-1)

    if unit:
        face_normals = torch.nn.functional.normalize(face_normals)
        # face_normals_length = face_normals.norm(dim=-1, keepdim=True)
        # face_normals = face_normals / (face_normals_length + 1e-10)

    return face_normals


def face_areas(vertices, faces):
    # args: vertices[N_v, 3], faces[N_f, 3]

    if faces.shape[-1] != 3:
        raise NotImplementedError("face_areas is only implemented for triangle meshes")
    faces_0, faces_1, faces_2 = torch.split(faces, 1, dim=1)
    face_v_0 = torch.index_select(vertices, 0, faces_0.reshape(-1))
    face_v_1 = torch.index_select(vertices, 0, faces_1.reshape(-1))
    face_v_2 = torch.index_select(vertices, 0, faces_2.reshape(-1))

    x1, x2, x3 = torch.split(face_v_0 - face_v_1, 1, dim=-1)
    y1, y2, y3 = torch.split(face_v_1 - face_v_2, 1, dim=-1)

    a = (x2 * y3 - x3 * y2) ** 2
    b = (x3 * y1 - x1 * y3) ** 2
    c = (x1 * y2 - x2 * y1) ** 2
    areas = torch.sqrt(a + b + c) * 0.5

    return areas.squeeze(-1)


######################################################################################
# Laplacian regularization using umbrella operator (Fujiwara / Desbrun).
# https://mgarland.org/class/geom04/material/smoothing.pdf
######################################################################################
def laplace_regularizer_const(v_pos, t_pos_idx):
    term = torch.zeros_like(v_pos)
    norm = torch.zeros_like(v_pos[..., 0:1])

    v0 = v_pos[t_pos_idx[:, 0], :]
    v1 = v_pos[t_pos_idx[:, 1], :]
    v2 = v_pos[t_pos_idx[:, 2], :]

    term.scatter_add_(0, t_pos_idx[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, t_pos_idx[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, t_pos_idx[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, t_pos_idx[:, 0:1], two)
    norm.scatter_add_(0, t_pos_idx[:, 1:2], two)
    norm.scatter_add_(0, t_pos_idx[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)


def numpy_to_ply(file_path, points):

    with open(file_path, 'w') as ply_file:
        # Write PLY header
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(points)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")

        # Write point data
        for point in points:
            ply_file.write(f"{point[0]} {point[1]} {point[2]}\n")
