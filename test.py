from fenics import *
from mshr import *
domain = Sphere(Point(0, 0, 0), 1)
mesh = generate_mesh(domain, 10)
# mesh = UnitCubeMesh(1,1,1)
V = FunctionSpace(mesh,"P",2)


expression = Expression("x[0]*x[0]+x[1]*x[1]+x[2]*x[2]",degree=1)
f=Function(V)
f.interpolate(expression)

import random
import numpy as np
cell = Cell(mesh,0)
vertices = cell.get_vertex_coordinates()
print("cell vertices:\n",np.array(vertices).reshape(4,3).transpose().reshape(12,))

for i in range(100000):
    mid_point = cell.midpoint()
    ran_point = Point(random.random(),random.random(),random.random())
    point = mid_point + ran_point
    if cell.contains(point):
        print(point.x(),point.y(),point.z())

element = V.element()
dofmap = V.dofmap()
print("dof_coordinates: ", element.tabulate_dof_coordinates(cell))
print("dofs: ", f.vector()[dofmap.cell_dofs(cell.index())])