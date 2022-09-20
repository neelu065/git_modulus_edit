from modulus.models.multiscale_fourier_net import MultiscaleFourierNetArch
import torch
import numpy as np
from modulus.key import Key


def make_dict(nr_layers):
    _dict = dict()
    names = [("weight", "weights"), ("bias", "biases"), ("weight_g", "alphas")]
    for i in range(nr_layers):
        for pt_name, tf_name in names:
            _dict["fc_layers." + str(i) + ".linear." + pt_name] = (
                "fc" + str(i) + "/" + tf_name + ":0"
            )
    for pt_name, tf_name in names[:2]:
        _dict["final_layer.linear." + pt_name] = "fc_final/" + tf_name + ":0"
    return _dict


def test_multiscale_fourier_net():
    filename = "./test_models/data/test_multiscale_fourier.npz"
    test_data = np.load(filename, allow_pickle=True)
    data_in = test_data["data_in"]
    Wbs = test_data["Wbs"][()]
    params = test_data["params"][()]
    frequency_1 = tuple(
        [test_data["frequency_1_name"][()]] + list(test_data["frequency_1_data"])
    )
    frequency_2 = tuple(
        [test_data["frequency_2_name"][()]] + list(test_data["frequency_2_data"])
    )
    frequencies = test_data["frequencies"]
    frequencies_params = test_data["frequencies_params"]
    # create graph
    arch = MultiscaleFourierNetArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        frequencies=(frequency_1, frequency_2),
        frequencies_params=(frequency_1, frequency_2),
        layer_size=params["layer_size"],
        nr_layers=params["nr_layers"],
    )
    name_dict = make_dict(params["nr_layers"])
    for _name, _tensor in arch.named_parameters():
        if _tensor.requires_grad:
            _tensor.data = torch.from_numpy(Wbs[name_dict[_name]].T)

    arch.fourier_layers_xyzt[0].frequencies = torch.from_numpy(
        Wbs["fourier_layer_xyzt_0:0"].T
    )
    arch.fourier_layers_xyzt[1].frequencies = torch.from_numpy(
        Wbs["fourier_layer_xyzt_1:0"].T
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


test_multiscale_fourier_net()
