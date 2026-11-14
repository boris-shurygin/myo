"""Render-only corner normals; usable by UE's Python without NumPy or MuJoCo."""

import math

NORMALS_STYLE = "angle75_v1"


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(value):
    length = math.sqrt(_dot(value, value))
    if length <= 0:
        raise ValueError("Cannot shade a degenerate triangle")
    return tuple(x / length for x in value)


def smooth_corner_normals(vertices, triangles):
    """Angle-weight coincident corners below 75 degrees, retaining sharp folds.

    Exact position grouping also handles STL/OBJ normal seams and triangle soup.
    Only attributes are produced; source positions, faces and winding stay intact.
    Opposite sides of a thin/open bone surface are never averaged together.
    """
    faces, incidents = [], {}
    for face_id, triangle in enumerate(triangles):
        points = [tuple(vertices[index]) for index in triangle]
        normal = _unit(_cross(_subtract(points[1], points[0]), _subtract(points[2], points[0])))
        faces.append(normal)
        for corner, point in enumerate(points):
            u = _subtract(points[(corner + 1) % 3], point)
            v = _subtract(points[(corner + 2) % 3], point)
            angle = math.atan2(math.sqrt(_dot(_cross(u, v), _cross(u, v))), _dot(u, v))
            incidents.setdefault(point, []).append((face_id, angle))
    threshold = math.cos(math.radians(75))
    normals = []
    for face_id, triangle in enumerate(triangles):
        for index in triangle:
            weighted = [0., 0., 0.]
            for neighbor, angle in incidents[tuple(vertices[index])]:
                if _dot(faces[face_id], faces[neighbor]) >= threshold:
                    for axis in range(3):
                        weighted[axis] += faces[neighbor][axis] * angle
            normals.append(_unit(weighted))
    return normals
