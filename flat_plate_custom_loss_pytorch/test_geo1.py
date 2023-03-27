from modulus.geometry.primitives_2d import Rectangle, Circle, Line
from modulus.utils.io.vtk import var_to_polyvtk
from modulus.geometry.parameterization import Parameterization, Parameter
import numpy as np

width = 0.6
height = 0.6

wall_left = Line((-width / 2, - height / 2), (- width / 2, height / 2), 1, parameterization=Parameterization({'alpha': (-10, 10)}))
wall_bottom = Line((- width / 2, - height / 2), (- width / 2, height / 2), 1).rotate(np.pi / 2)

wall_right = Line((width / 2, - height / 2), (width / 2, height / 2), -1)
wall_top = Line((width / 2, - height / 2), (width / 2, height / 2), -1).rotate(np.pi / 2)

# make plate with parameterized hole
# make parameterized primitives
# plate = Rectangle(point_1=(-1, -1), point_2=(1, 1))
# # y_pos = Parameter("y_pos")
# # parameterization = Parameterization({y_pos: (-1, 1)})
# # circle = Circle(center=(0, y_pos), radius=0.3, parameterization=parameterization)
# plate2 = Line(point_1=(0, 0), point_2=(0, 0.1)).rotate(np.pi / 2)
# # plate2 = Rectangle(point_1=(-0.1, -1e-10), point_2=(0.1, 1e-10))
# geo = plate - plate2
#
# abc = plate2.sample_boundary(5000)
# var_to_polyvtk(abc, "plate2")
wall = [wall_left, wall_bottom, wall_top, wall_right]
print("hi")
left = wall_left.sample_boundary(500)
var_to_polyvtk(left, "wall_left")
right = wall_right.sample_boundary(500)
var_to_polyvtk(right, "wall_r")
top = wall_top.sample_boundary(500)
var_to_polyvtk(top, "wall_top")
bottom = wall_bottom.sample_boundary(500)
var_to_polyvtk(bottom, "wall_bott")
# sample geometry over entire parameter range
# s = geo.sample_boundary(nr_points=100000)
# var_to_polyvtk(s, "parameterized_boundary")
# s = geo.sample_interior(nr_points=100000)
# var_to_polyvtk(s, "parameterized_interior")
#
# # sample specific parameter
# s = geo.sample_boundary(
#     nr_points=100000) #, parameterization=Parameterization({y_pos: 0})
# #)
# var_to_polyvtk(s, "y_pos_zero_boundary")
# s = geo.sample_interior(
#     nr_points=100000) #, parameterization=Parameterization({y_pos: 0})
# #)
# var_to_polyvtk(s, "y_pos_zero_interior")
