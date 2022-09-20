import torch
from torch import Tensor
from typing import Dict, List, Tuple

import modulus.models.fully_connected as fully_connected
import modulus.models.layers.layers as layers
from modulus.models.layers.layers import Activation
from .arch import Arch
from modulus.key import Key


class FourierNetArch(Arch):
    """Fourier encoding fully-connected neural network.

    This network is a fully-connected neural network that encodes the input features
    into Fourier space using sinesoidal activation functions. This helps reduce spectal
    bias during training.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list.
    output_keys : List[Key]
        Output key list.
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    frequencies : Tuple, optional
        A tuple that describes the Fourier encodings to use any inputs in the list
        `['x', 'y', 'z', 't']`.
        The first element describes the type of frequency encoding
        with options, `'gaussian', 'full', 'axis', 'diagonal'`, by default
        ("axis", [i for i in range(10)])

        :obj:`'gaussian'` samples frequency of Fourier series from Gaussian.

        :obj:`'axis'` samples along axis of spectral space with the given list range of
        frequencies.

        :obj:`'diagonal'` samples along diagonal of spectral space with the given list range
        of frequencies.

        :obj:`'full'` samples along entire spectral space for all combinations of frequencies
        in given list.

    frequencies_params : Tuple, optional
        Same as `frequencies` used for encodings of any inputs not in the list
        `['x', 'y', 'z', 't']`.
        By default ("axis", [i for i in range(10)])
    activation_fn : Activation, optional
        Activation function, by default :obj:`Activation.SILU`
    layer_size : int, optional
        Layer size for every hidden layer of the model, by default 512
    nr_layers : int, optional
        Number of hidden layers of the model, by default 6
    skip_connections : bool, optional
        Apply skip connections every 2 hidden layers, by default False
    weight_norm : bool, optional
        Use weight norm on fully connected layers, by default True
    adaptive_activations : bool, optional
        Use an adaptive activation functions, by default False

    Variable Shape
    --------------
    - Input variable tensor shape: :math:`[N, size]`
    - Output variable tensor shape: :math:`[N, size]`

    Example
    -------
    Gaussian frequencies

    >>> std = 1.0; num_freq = 10
    >>> model = .fourier_net.FourierNetArch(
    >>>    [Key("x", size=2)],
    >>>    [Key("y", size=2)],
    >>>    frequencies=("gaussian", std, num_freq))

    Diagonal frequencies

    >>> frequencies = [1.0, 2.0, 3.0, 4.0]
    >>> model = .fourier_net.FourierNetArch(
    >>>    [Key("x", size=2)],
    >>>    [Key("y", size=2)],
    >>>    frequencies=("diagonal", frequencies))

    Full frequencies

    >>> frequencies = [1.0, 2.0, 3.0, 4.0]
    >>> model = .fourier_net.FourierNetArch(
    >>>    [Key("x", size=2)],
    >>>    [Key("y", size=2)],
    >>>    frequencies=("full", frequencies))

    Note
    ----
    For information regarding adaptive activations please refer to
    https://arxiv.org/abs/1906.01170.
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        frequencies: Tuple = ("axis", [i for i in range(10)]),
        frequencies_params: Tuple = ("axis", [i for i in range(10)]),
        activation_fn: Activation = Activation.SILU,
        layer_size: int = 512,
        nr_layers: int = 6,
        skip_connections: bool = False,
        weight_norm: bool = True,
        adaptive_activations: bool = False,
    ) -> None:
        super().__init__(
            input_keys=input_keys, output_keys=output_keys, detach_keys=detach_keys
        )

        if frequencies_params is None:
            frequencies_params = frequencies

        self.xyzt_var = [x for x in self.input_key_dict if x in ["x", "y", "z", "t"]]
        self.params_var = [
            x for x in self.input_key_dict if x not in ["x", "y", "z", "t"]
        ]

        in_features_xyzt = sum(
            (v for k, v in self.input_key_dict.items() if k in self.xyzt_var)
        )
        in_features_params = sum(
            (v for k, v in self.input_key_dict.items() if k in self.params_var)
        )
        in_features = in_features_xyzt + in_features_params
        out_features = sum(self.output_key_dict.values())

        if in_features_xyzt > 0:
            self.fourier_layer_xyzt = layers.FourierLayer(
                in_features=in_features_xyzt, frequencies=frequencies
            )
            in_features += self.fourier_layer_xyzt.out_features()
        else:
            self.fourier_layer_xyzt = None

        if in_features_params > 0:
            self.fourier_layer_params = layers.FourierLayer(
                in_features=in_features_params, frequencies=frequencies_params
            )
            in_features += self.fourier_layer_params.out_features()
        else:
            self.fourier_layer_params = None

        self.fc = fully_connected.FullyConnectedArchCore(
            in_features=in_features,
            layer_size=layer_size,
            out_features=out_features,
            nr_layers=nr_layers,
            skip_connections=skip_connections,
            activation_fn=activation_fn,
            adaptive_activations=adaptive_activations,
            weight_norm=weight_norm,
        )

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.prepare_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
        )

        if self.fourier_layer_xyzt is not None:
            in_xyzt_var = self.prepare_input(
                in_vars,
                self.xyzt_var,
                detach_dict=self.detach_key_dict,
                dim=-1,
                input_scales=self.input_scales,
            )
            fourier_xyzt = self.fourier_layer_xyzt(in_xyzt_var)
            x = torch.cat((x, fourier_xyzt), dim=-1)

        if self.fourier_layer_params is not None:
            in_params_var = self.prepare_input(
                in_vars,
                self.params_var,
                detach_dict=self.detach_key_dict,
                dim=-1,
                input_scales=self.input_scales,
            )
            fourier_params = self.fourier_layer_params(in_params_var)
            x = torch.cat((x, fourier_params), dim=-1)

        x = self.fc(x)
        return self.prepare_output(
            x, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )
