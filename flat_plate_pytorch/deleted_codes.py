# add validator
# mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
# openfoam_var = csv_to_dict(
#     to_absolute_path("openfoam/cavity_uniformVel0.csv"), mapping
# )
# openfoam_var["x"] += -width / 2  # center OpenFoam data
# openfoam_var["y"] += -height / 2  # center OpenFoam data
# openfoam_invar_numpy = {
#     key: value for key, value in openfoam_var.items() if key in ["x", "y"]
# }
# openfoam_outvar_numpy = {
#     key: value for key, value in openfoam_var.items() if key in ["u", "v"]
# }
# openfoam_validator = PointwiseValidator(
#     nodes=nodes,
#     invar=openfoam_invar_numpy,
#     true_outvar=openfoam_outvar_numpy,
#     batch_size=1024,
#     plotter=ValidatorPlotter(),
# )
# ldc_domain.add_validator(openfoam_validator)
# # -----------------------------
# # top wall
# top_wall = PointwiseBoundaryConstraint(
#     nodes=nodes,
#     geometry=rec,
#     outvar={"u": 1.0, "v": 0},
#     batch_size=cfg.batch_size.TopWall,
#     lambda_weighting={"u": 1.0 - 20 * Abs(x), "v": 1.0},  # weight edges to be zero
#     criteria=Eq(y, cfg.custom.domain_height / 2),
# )
# ldc_domain.add_constraint(top_wall, "top_wall")
#
# # no slip
# no_slip = PointwiseBoundaryConstraint(
#     nodes=nodes,
#     geometry=geo,
#     outvar={"u": 0, "v": 0},
#     batch_size=cfg.batch_size.NoSlip,
#     criteria=y < cfg.custom.domain_height / 2,
# )
# ldc_domain.add_constraint(no_slip, "no_slip")
#
# # interior
# interior = PointwiseInteriorConstraint(
#     nodes=nodes,
#     geometry=rec,
#     outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
#     batch_size=cfg.batch_size.Interior,
#     lambda_weighting={
#         "continuity": Symbol("sdf"),
#         "momentum_x": Symbol("sdf"),
#         "momentum_y": Symbol("sdf"),
#     },
# )
# ldc_domain.add_constraint(interior, "interior")


## Codes from channel flow

# simulation params
channel_length = (-10.0, 30.0)
channel_width = (-10.0, 10.0)
cylinder_center = (0.0, 0.0)
cylinder_radius = 0.5
inlet_vel = 1.0
# define sympy variables to parametrize domain curves
# x, y = Symbol("x"), Symbol("y")

# define geometry
channel = Channel2D(
    (channel_length[0], channel_width[0]), (channel_length[1], channel_width[1])
)
inlet = Line(
    (channel_length[0], channel_width[0]),
    (channel_length[0], channel_width[1]),
    normal=1,
)
outlet = Line(
    (channel_length[1], channel_width[0]),
    (channel_length[1], channel_width[1]),
    normal=1,
)
wall_top = Line(
    (channel_length[1], channel_width[0]),
    (channel_length[1], channel_width[1]),
    normal=1,
)
cylinder = Circle(cylinder_center, cylinder_radius)
volume_geo = channel - cylinder


# ---- create_geometry codes ---- #
# rec = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))

# geo = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))
# obstacle_above = Line((0, 0), (0, cfg.custom.obstacle_length), 1)
# obstacle_below = Line((0, 0), (0, cfg.custom.obstacle_length), 1)
#
# wake1_above = Line((0, -1 * cfg.custom.obstacle_length), (0, 0), 1)  # Wake to enforce kutta condition
# wake2_above = Line((0, -2 * cfg.custom.obstacle_length), (0, -1 * cfg.custom.obstacle_length),
#                    1)  # Wake to enforce kutta condition
# wake3_above = Line((0, -3 * cfg.custom.obstacle_length), (0, -2 * cfg.custom.obstacle_length),
#                    1)  # Wake to enforce kutta condition
#
# wake1_below = Line((0, -1 * cfg.custom.obstacle_length), (0, 0), 1)  # Wake to enforce kutta condition
# wake2_below = Line((0, -2 * cfg.custom.obstacle_length), (0, -1 * cfg.custom.obstacle_length),
#                    1)  # Wake to enforce kutta condition
# wake3_below = Line((0, -3 * cfg.custom.obstacle_length), (0, -2 * cfg.custom.obstacle_length),
#                    1)  # Wake to enforce kutta condition
#
# obstacle_above.rotate(np.pi / 2)
# obstacle_below.rotate(np.pi / 2)
#
# wake1_above.rotate(np.pi / 2)
# wake2_above.rotate(np.pi / 2)
# wake3_above.rotate(np.pi / 2)
#
# wake1_below.rotate(np.pi / 2)
# wake2_below.rotate(np.pi / 2)
# wake3_below.rotate(np.pi / 2)

# add monitor
monitor = PointwiseMonitor(
    chip2d.sample_boundary(10000, criteria=Eq(y, source_origin[1])),
    output_names=["theta_s"],
    metrics={
        "peak_temp": lambda var: torch.max(var["theta_s"]),
    },
    nodes=nodes,
)
domain.add_monitor(monitor)

#
# inlet = PointwiseBoundaryConstraint(
#     nodes=nodes,
#     geometry=wake1_above,
#     outvar={"u": 1},
#     batch_size=cfg.batch_size.TopWall,
# )
# ldc_domain.add_constraint(inlet, "inlet")
