from modulus.geometry.primitives_2d import Rectangle, Line
import numpy as np
from modulus.hydra import ModulusConfig


def create_geometry(cfg: ModulusConfig):
    height = cfg.custom.unscaled_domain_height * cfg.custom.obstacle_length  # Domain height
    width = cfg.custom.unscaled_domain_width * cfg.custom.obstacle_length  # Domain width

    geo = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))
    obstacle_above = Line((0, 0), (0, cfg.custom.obstacle_length), 1)
    obstacle_below = Line((0, 0), (0, cfg.custom.obstacle_length), 1)

    wake1_above = Line((0, -1 * cfg.custom.obstacle_length), (0, 0), 1)  # Wake to enforce kutta condition
    wake2_above = Line((0, -2 * cfg.custom.obstacle_length), (0, -1 * cfg.custom.obstacle_length),
                       1)  # Wake to enforce kutta condition
    wake3_above = Line((0, -3 * cfg.custom.obstacle_length), (0, -2 * cfg.custom.obstacle_length),
                       1)  # Wake to enforce kutta condition

    wake1_below = Line((0, -1 * cfg.custom.obstacle_length), (0, 0), 1)  # Wake to enforce kutta condition
    wake2_below = Line((0, -2 * cfg.custom.obstacle_length), (0, -1 * cfg.custom.obstacle_length),
                       1)  # Wake to enforce kutta condition
    wake3_below = Line((0, -3 * cfg.custom.obstacle_length), (0, -2 * cfg.custom.obstacle_length),
                       1)  # Wake to enforce kutta condition

    obs_above = obstacle_above.rotate(np.pi / 2)
    obs_below = obstacle_below.rotate(np.pi / 2)

    wk1_above = wake1_above.rotate(np.pi / 2)
    wk2_above = wake2_above.rotate(np.pi / 2)
    wk3_above = wake3_above.rotate(np.pi / 2)

    wk1_below = wake1_below.rotate(np.pi / 2)
    wk2_below = wake2_below.rotate(np.pi / 2)
    wk3_below = wake3_below.rotate(np.pi / 2)

    return geo, obs_above, obs_below, wk1_above, wk1_below, wk2_above, wk2_below, wk3_above, wk3_below
