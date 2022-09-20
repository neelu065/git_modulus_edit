import torch
from modulus.loss import (
    PointwiseLossNorm,
    DecayedPointwiseLossNorm,
    IntegralLossNorm,
    DecayedIntegralLossNorm,
)


def test_loss_norm():
    # make pointwise test values
    invar = {"x": torch.arange(10)[:, None], "area": torch.ones(10)[:, None] / 10}
    pred_outvar = {"u": torch.arange(10)[:, None]}
    true_outvar = {"u": torch.arange(10)[:, None] + 2}
    lambda_weighting = {"u": torch.ones(10)[:, None]}

    # Test Pointwise l2
    loss = PointwiseLossNorm(2)
    l = loss.forward(invar, pred_outvar, true_outvar, lambda_weighting, step=0)
    assert torch.isclose(l["u"], torch.tensor(4.0))

    # Test Pointwise l1
    loss = PointwiseLossNorm(1)
    l = loss.forward(invar, pred_outvar, true_outvar, lambda_weighting, step=0)
    assert torch.isclose(l["u"], torch.tensor(2.0))

    # Test Decayed Pointwise l2
    loss = DecayedPointwiseLossNorm(2, 1, decay_steps=1000, decay_rate=0.5)
    l = loss.forward(invar, pred_outvar, true_outvar, lambda_weighting, step=0)
    assert torch.isclose(l["u"], torch.tensor(4.0))
    l = loss.forward(invar, pred_outvar, true_outvar, lambda_weighting, step=1000)
    assert torch.isclose(l["u"], torch.tensor(2.82842712))
    l = loss.forward(invar, pred_outvar, true_outvar, lambda_weighting, step=1000000)
    assert torch.isclose(l["u"], torch.tensor(2.0))

    # make Integral test values
    list_invar = [
        {"x": torch.arange(10)[:, None], "area": torch.ones(10)[:, None] / 10}
    ]
    list_pred_outvar = [{"u": torch.arange(10)[:, None]}]
    list_true_outvar = [{"u": torch.tensor(2.5)[None, None]}]
    list_lambda_weighting = [{"u": torch.ones(1)[None, None]}]

    # Test Integral l2
    loss = IntegralLossNorm(2)
    l = loss.forward(
        list_invar, list_pred_outvar, list_true_outvar, list_lambda_weighting, step=0
    )
    assert torch.isclose(l["u"], torch.tensor(4.0))

    # Test Integral l1
    loss = IntegralLossNorm(1)
    l = loss.forward(
        list_invar, list_pred_outvar, list_true_outvar, list_lambda_weighting, step=0
    )
    assert torch.isclose(l["u"], torch.tensor(2.0))

    # Test Decayed Integral l2
    loss = DecayedIntegralLossNorm(2, 1, decay_steps=1000, decay_rate=0.5)
    l = loss.forward(
        list_invar, list_pred_outvar, list_true_outvar, list_lambda_weighting, step=0
    )
    assert torch.isclose(l["u"], torch.tensor(4.0))
    l = loss.forward(
        list_invar, list_pred_outvar, list_true_outvar, list_lambda_weighting, step=1000
    )
    assert torch.isclose(l["u"], torch.tensor(2.82842712))
    l = loss.forward(
        list_invar,
        list_pred_outvar,
        list_true_outvar,
        list_lambda_weighting,
        step=1000000,
    )
    assert torch.isclose(l["u"], torch.tensor(2.0))
