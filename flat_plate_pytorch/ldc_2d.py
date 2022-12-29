from sympy import Symbol, Eq, Abs, cos, sin
import torch
import modulus
import numpy as np
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry import Bounds

from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.node import Node
from modulus.domain.inferencer import PointwiseInferencer
from modulus.key import Key
from modulus.eq.pdes.navier_stokes import NavierStokes
from modulus.utils.io import InferencerPlotter
from modulus.geometry.primitives_2d import Rectangle, Line, Circle, Channel2D

# misc files imports
from geometry_create import create_geometry
from poison_2d import Poison_2D


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # ----------- Poisson 2D ----------- #
    poisson_2d = Poison_2D(cfg)
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("alpha")],
        output_keys=[Key("u"), Key("v"), Key("phi")],
        cfg=cfg.arch.fully_connected  # six hidden layers with 512 neurons per layer.
    )
    nodes = poisson_2d.make_nodes() + [flow_net.make_node(name="flow_network")]
    # ----------- Poisson 2D ----------- #

    # ----------- navier stokes ----------------- #
    # ns = NavierStokes(nu=0.01, rho=1.0, dim=2, time=False)
    # flow_net = instantiate_arch(
    #     input_keys=[Key("x"), Key("y")],
    #     output_keys=[Key("u"), Key("v"), Key("p")],
    #     cfg=cfg.arch.fully_connected,  # six hidden layers with 512 neurons per layer.
    # )
    # nodes = ns.make_nodes() + [flow_net.make_node(name="flow_network")]
    # ----------- navier stokes ----------------- #

    # domain height and weight
    height = cfg.custom.unscaled_domain_height * cfg.custom.obstacle_length
    width = cfg.custom.unscaled_domain_width * cfg.custom.obstacle_length

    # make geometry
    x, y, alpha = Symbol('x'), Symbol('y'), Symbol('alpha')
    geo, obstacle_above, obstacle_below, wake1_above, wake1_below, wake2_above, \
    wake2_below, wake3_above, wake3_below = create_geometry(cfg)

    # make ldc domain
    ldc_domain = Domain()

    alpha_range = {
        alpha: lambda batch_size: np.full((batch_size, 1),
                                          np.random.uniform(- np.pi * cfg.custom.free_stream_velocity / 180,
                                                            np.pi * cfg.custom.free_stream_velocity / 180))}

    u_x = cfg.custom.free_stream_velocity * cos(alpha)  # 10 * cos(alpha) # using np please verify
    u_y = cfg.custom.free_stream_velocity * sin(alpha)  # 10 * sin(alpha)

    # Constraints defined #
    # add constraints to solver
    leftWall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        batch_size=cfg.batch_size.LeftWall,
        outvar={"residual_u_comp": 0, "residual_v_comp": 0},
        criteria=~Eq(x, -width / 2),  # As the left wall lies on x = -width/2, we set the criteria to be x = -width/2
        parameterization=alpha_range,
        # Unhashed error if used another dictionary: {alpha_range} instead use alpha_range or remove {}.
        fixed_dataset=False
    )
    ldc_domain.add_constraint(leftWall, name="LeftWall")

    topWall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        batch_size=cfg.batch_size.TopWall,
        outvar={"residual_u_comp": 0, "residual_v_comp": 0},
        # Mimicing the far field conditions "u":u_x , "v": u_y,
        criteria=~Eq(y, height / 2),  # As the top wall lies on y = height/2, we set the criteria to be y = height/2
        parameterization=alpha_range,
        fixed_dataset=False
    )
    ldc_domain.add_constraint(topWall, name="TopWall")
    #
    rightWall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        batch_size=cfg.batch_size.RightWall,
        outvar={"residual_u_comp": 0, "residual_v_comp": 0},
        # Mimicing the far field conditions "u":u_x , "v": u_y,
        criteria=~Eq(x, width / 2),  # As the right wall lies on x = width/2, we set the criteria to be x = width/2
        parameterization=alpha_range,
        fixed_dataset=False
    )
    ldc_domain.add_constraint(rightWall, name="RightWall")
    #
    bottomWall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        batch_size=cfg.batch_size.BottomWall,
        outvar={"residual_u_comp": 0, "residual_v_comp": 0},
        # Mimicing the far field conditions "u":u_x , "v": u_y,
        criteria=~Eq(y, -height / 2),
        # As the bottom wall lies on y = -height/2, we set the criteria to be y = -height/2
        parameterization=alpha_range,
        fixed_dataset=False
    )
    ldc_domain.add_constraint(bottomWall, name="BottomWall")

    obstacleLineAbove = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=obstacle_above,
        batch_size=cfg.batch_size.obstacle_above,
        outvar={"residual_u": u_x, 'residual_obstacle_above': 0},  # u_x changed to residual_u
        # Setting up the no slip condition for the obstacle.
        lambda_weighting={"residual_u": 100, "residual_obstacle_above": 100},  # Symbol("sdf")},
        # check Symbol("sdf") --> geo.sdf # Weights for the loss function.
        parameterization=alpha_range,
        fixed_dataset=False
    )
    ldc_domain.add_constraint(obstacleLineAbove, name="obstacleLineAbove")

    obstacleLineBelow = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=obstacle_below,
        batch_size=cfg.batch_size.obstacle_below,
        outvar={"u": u_x, 'residual_obstacle_below': 0},
        lambda_weighting={"u": 100, "residual_obstacle_below": 100},
        parameterization=alpha_range,
        fixed_dataset=False
    )
    ldc_domain.add_constraint(obstacleLineBelow, name="obstacleLineBelow")

    l = lambda x: x / (3 * cfg.custom.obstacle_length)  # x = 0 at the trailing edge of the obstacle
    wakeLine1_Above = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wake1_above,
        batch_size=cfg.batch_size.wake1_above,
        outvar={"u": u_x, "v": u_y * l(x)},
        lambda_weighting={"u": 100, "v": 100},
        parameterization=alpha_range,
        fixed_dataset=False
    )
    ldc_domain.add_constraint(wakeLine1_Above, name="wakeLine1_Above")
    #
    wakeLine2_Above = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wake2_above,
        batch_size=cfg.batch_size.wake2_above,
        outvar={"u": u_x, "v": u_y * l(x)},
        lambda_weighting={"u": 100, "v": 100},
        parameterization=alpha_range,
        fixed_dataset=False
    )
    ldc_domain.add_constraint(wakeLine2_Above, name="wakeLine2_Above")
    #
    wakeLine3_Above = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wake3_above,
        batch_size=cfg.batch_size.wake3_above,
        outvar={"u": u_x, "v": u_y * l(x)},
        lambda_weighting={"u": 100, "v": 100},
        parameterization=alpha_range,
        fixed_dataset=False
    )

    ldc_domain.add_constraint(wakeLine3_Above, name="wakeLine3_Above")
    #
    wakeLine1_Below = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wake1_below,
        batch_size=cfg.batch_size.wake1_below,  # batch_size=150 * 2
        outvar={"u": u_x, "v": u_y * l(x)},
        lambda_weighting={"u": 100, "v": 100},
        parameterization=alpha_range,
        fixed_dataset=False
    )
    ldc_domain.add_constraint(wakeLine1_Below, name="wakeLine1_Below")

    wakeLine2_Below = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wake2_below,
        outvar={"u": u_x, "v": u_y * l(x)},
        batch_size=cfg.batch_size.wake2_below,  # batch_size=150 * 2
        lambda_weighting={"u": 100, "v": 100},
        parameterization=alpha_range,
        fixed_dataset=False
    )
    ldc_domain.add_constraint(wakeLine2_Below, name="wakeLine2_Below")

    wakeLine3_Below = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wake3_below,
        outvar={"u": u_x, "v": u_y * l(x)},
        batch_size=cfg.batch_size.wake1_above,  # batch_size=150 * 2
        lambda_weighting={"u": 100, "v": 100},
        parameterization=alpha_range,
        fixed_dataset=False
    )
    ldc_domain.add_constraint(wakeLine3_Below, name="wakeLine3_Below")

    interior = PointwiseInteriorConstraint(
        geometry=geo,
        nodes=nodes,
        outvar={"Poisson_2D": 0, "residual_u": 0, "residual_v": 0},
        bounds=geo.bounds.bound_ranges,
        lambda_weighting={
            "Poisson_2D": Symbol("sdf"),
            "residual_u": Symbol("sdf"),
            "residual_v": Symbol("sdf"),
        },
        parameterization=alpha_range,
        fixed_dataset=False,
        batch_size=cfg.batch_size.Interior
    )

    ldc_domain.add_constraint(interior, name="interior")
    # Constraints defined #

    # add inference data

    # ----- Inference ----- #
    # mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
    # openfoam_var = csv_to_dict(
    #     to_absolute_path("openfoam/cylinder_nu_0.020.csv"), mapping
    # )
    # openfoam_invar_numpy = {
    #     key: value for key, value in openfoam_var.items() if key in ["x", "y"]
    # }
    # grid_inference = PointwiseInferencer(
    #     nodes=nodes,
    #     invar=openfoam_invar_numpy,
    #     output_names=["u", "v", "p"],
    #     batch_size=1024,
    #     plotter=InferencerPlotter(),
    # )
    # ldc_domain.add_inferencer(grid_inference, "inf_data")
    # ----- Inference ----- #

    # ----- Validator ----- #
    # openfoam_validator = PointwiseValidator(
    #     nodes=nodes, invar=openfoam_invar_numpy, true_outvar=openfoam_outvar_numpy
    # )
    # domain.add_validator(openfoam_validator)
    # ----- Validator ----- #
    # make solver
    slv = Solver(cfg, ldc_domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()