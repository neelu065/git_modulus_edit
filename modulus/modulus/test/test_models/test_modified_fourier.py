from modulus.models.modified_fourier_net import ModifiedFourierNetArch
import torch
import numpy as np
from modulus.key import Key


def make_dict(nr_layers):
    _dict = dict()
    names = [("weight", "weights"), ("bias", "biases"), ("weight_g", "alphas")]
    for layer_name in ("fc_u", "fc_v"):
        for pt_name, tf_name in names:
            _dict[layer_name + ".linear." + pt_name] = layer_name + "/" + tf_name + ":0"
    for i in range(nr_layers):
        for pt_name, tf_name in names:
            if i == 0:
                _dict["fc_" + str(i) + ".linear." + pt_name] = (
                    "fc" + str(i) + "/" + tf_name + ":0"
                )
            else:
                _dict["fc_layers." + str(i - 1) + ".linear." + pt_name] = (
                    "fc" + str(i) + "/" + tf_name + ":0"
                )
    for pt_name, tf_name in names[:2]:
        _dict["final_layer.linear." + pt_name] = "fc_final/" + tf_name + ":0"
    return _dict


def test_modified_fourier_net():
    filename = "./test_models/data/test_modified_fourier.npz"
    test_data = np.load(filename, allow_pickle=True)
    data_in = test_data["data_in"]
    Wbs = test_data["Wbs"][()]
    params = test_data["params"][()]
    frequencies = test_data["frequencies"]
    frequencies_params = test_data["frequencies_params"]
    # create graph
    arch = ModifiedFourierNetArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        frequencies=("axis,diagonal", frequencies),
        frequencies_params=("axis,diagonal", frequencies_params),
        layer_size=params["layer_size"],
        nr_layers=params["nr_layers"],
    )
    name_dict = make_dict(params["nr_layers"])
    for _name, _tensor in arch.named_parameters():
        if _tensor.requires_grad:
            _tensor.data = torch.from_numpy(Wbs[name_dict[_name]].T)

    arch.fourier_layer_xyzt.frequencies = torch.from_numpy(
        Wbs["fourier_layer_xyzt:0"].T
    )
    data_out2 = arch(
        {"x": torch.from_numpy(data_in[:, 0:1]), "y": torch.from_numpy(data_in[:, 1:2])}
    )
    data_out2 = data_out2["u"].detach().numpy()
    # load outputs
    data_out1 = test_data["data_out"]
    # verify
    assert np.allclose(data_out1, data_out2, rtol=1e-3), "Test failed!"
    print("Success!")


test_modified_fourier_net()