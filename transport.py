import torch
import enum
import numpy as np

def expand_t_like_x(t, x):
    """
    Reshapes time t for broadcasting to the same shape as x.
    """
    dims = [1] * (len(x.size()) - 1)
    return t.view(t.size(0), *dims)

class PathType(enum.Enum):
    """
    Enum for different types of interpolation paths.
    """
    LINEAR = enum.auto()

class ModelType(enum.Enum):
    """
    Enum for the model's prediction target.
    """
    VELOCITY = enum.auto()

class ICPlan:
    """
    Implements the Linear Interpolation Path (Independent Coupling Plan).
    This path defines a straight line between a noise sample (x0) and a data sample (x1).
    """
    def compute_xt(self, t, x0, x1):
        """
        Computes the interpolated sample xt at time t.
        xt = (1-t) * x0 + t * x1
        """
        t_expanded = expand_t_like_x(t, x1)
        return (1 - t_expanded) * x0 + t_expanded * x1

    def compute_ut(self, t, x0, x1, xt=None):
        """
        Computes the vector field (velocity) ut for the linear path.
        For a linear path, the velocity is constant: u_t = x1 - x0.
        """
        return x1 - x0

    def plan(self, t, x0, x1):
        """
        Generates the training tuple (t, xt, ut).
        """
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut

class Transport:
    """
    Main class to handle the Flow Matching logic, inspired by the reference implementation.
    """
    def __init__(self, *, path_type='LINEAR', prediction='VELOCITY', train_eps=1e-5):
        path_options = {
            PathType.LINEAR: ICPlan,
        }
        self.path_sampler = path_options[PathType[path_type]]()
        self.model_type = ModelType[prediction]
        self.train_eps = train_eps

    def sample_time(self, num_samples, device):
        """
        Samples time steps t uniformly from [train_eps, 1 - train_eps].
        """
        t0 = self.train_eps
        t1 = 1.0
        t = torch.rand((num_samples,), device=device) * (t1 - t0) + t0
        return t

    def training_losses(self, model, x1, model_kwargs=None):
        """
        Computes the training loss for a given model and data batch.
        """
        if model_kwargs is None:
            model_kwargs = {}

        # 1. Sample noise x0 and time t
        x0 = torch.randn_like(x1)
        t = self.sample_time(x1.shape[0], x1.device)

        # 2. Compute the interpolated sample xt and the target velocity ut
        _, xt, ut = self.path_sampler.plan(t, x0, x1)
        
        # 3. Get model prediction
        # The original code scales time to [0, 999] for the embedding layer.
        time_input = t * 999.0
        
        model_output = model(xt, time_input, **model_kwargs)

        # 4. Return predictions and targets for flexible loss computation
        if self.model_type == ModelType.VELOCITY:
            loss = torch.nn.functional.mse_loss(model_output, ut)
            return {'loss': loss, 'pred': model_output, 'target': ut}
        else:
            # This part can be expanded to support other model types like NOISE or SCORE
            raise NotImplementedError(f"Model type {self.model_type} not implemented yet.")