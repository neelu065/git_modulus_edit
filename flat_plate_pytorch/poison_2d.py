from modulus.eq.pde import PDE
from sympy import Symbol, Function
from sympy import sin, cos
from modulus.hydra import ModulusConfig


class Poison_2D(PDE):  # phi changed to self.phi
    name = "Poisson_2D"

    # @hydra.main(config_path="conf", config_name="config")
    def __init__(self, cfg: ModulusConfig):

        x, y, alpha = Symbol("x"), Symbol("y"), Symbol("alpha")
        input_variables = {"x": x, "y": y, "alpha": alpha}

        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        phi = Function("phi")(*input_variables)

        self.equations = {}
        self.equations["residual_u"] = u - phi.diff(x)
        self.equations["residual_v"] = v - phi.diff(y)

        # For the far field conditions, we need to define the boundary conditions for the velocity components
        self.equations["residual_u_comp"] = u - cfg.custom.free_stream_velocity * cos(alpha)
        self.equations["residual_v_comp"] = v - cfg.custom.free_stream_velocity * sin(alpha)
        self.equations["residual_obstacle_above"] = v
        self.equations["residual_obstacle_below"] = v
        self.equations["residual_obstacle_wake1_above"] = v - cfg.custom.free_stream_velocity*sin(alpha)*x/(3*cfg.custom.obstacle_length)
        self.equations["residual_obstacle_wake2_above"] = v - cfg.custom.free_stream_velocity*sin(alpha)*x/(3*cfg.custom.obstacle_length)
        self.equations["residual_obstacle_wake3_above"] = v - cfg.custom.free_stream_velocity*sin(alpha)*x/(3*cfg.custom.obstacle_length)
        self.equations["residual_obstacle_wake1_below"] = v - cfg.custom.free_stream_velocity*sin(alpha)*x/(3*cfg.custom.obstacle_length)
        self.equations["residual_obstacle_wake2_below"] = v - cfg.custom.free_stream_velocity*sin(alpha)*x/(3*cfg.custom.obstacle_length)
        self.equations["residual_obstacle_wake3_below"] = v - cfg.custom.free_stream_velocity*sin(alpha)*x/(3*cfg.custom.obstacle_length)
        self.equations["Poisson_2D"] = (phi.diff(x)).diff(x) + (phi.diff(y)).diff(y)  # grad^2(phi)
