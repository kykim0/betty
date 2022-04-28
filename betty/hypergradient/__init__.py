from .darts import darts
from .cg import cg
from .default import default
from .neumann import neumann
from .reinforce import reinforce

grad_fn_mapping = {
    'default': default,
    'darts': darts,
    'neumann': neumann,
    'cg': cg,
    'reinforce': reinforce
}

def get_grad_fn(grad_fn_type):
    assert grad_fn_type in grad_fn_mapping
    return grad_fn_mapping[grad_fn_type]
