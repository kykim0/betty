import torch

from betty.utils import to_vec


def darts(vector, curr, prev):
    """
    Approximate the matrix-vector multiplication with the best response Jacobian by the
    finite difference method. More specifically, we modified the finite difference method proposed
    in `DARTS: Differentiable Architecture Search <https://arxiv.org/pdf/1806.09055.pdf>`_ by
    re-interpreting it from the implicit differentiation perspective. Empirically, this method
    achieves better memory efficiency, training wall time, and test accuracy that other methods.
    """
    config = curr.config
    R = config.darts_alpha
    eps = R / to_vec(vector).norm()

    # positive
    for p, v in zip(curr.trainable_parameters(), vector):
        p.data.add_(v.data, alpha=eps)
    loss_p = curr.training_step(curr.cur_batch)
    grad_p = torch.autograd.grad(loss_p, prev.trainable_parameters())

    # negative
    for p, v in zip(curr.trainable_parameters(), vector):
        p.data.add_(v.data, alpha=-2*eps)
    loss_n = curr.training_step(curr.cur_batch)
    grad_n = torch.autograd.grad(loss_n, prev.trainable_parameters())

    # reverse weight change
    for p, v in zip(curr.trainable_parameters(), vector):
        p.data.add(v.data, alpha=eps)

    implicit_grad = [(x - y).div_(2 * eps) for x, y in zip(grad_n, grad_p)]

    return implicit_grad
