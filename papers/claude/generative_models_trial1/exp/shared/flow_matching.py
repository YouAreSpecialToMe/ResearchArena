"""Flow matching utilities: OT-CFM interpolation, ODE solver, sampling."""
import torch


def ot_cfm_sample_t_and_xt(x0, noise=None):
    """Sample (t, x_t, target_v) for OT-CFM training.

    x_t = (1-t)*x_0 + t*noise, velocity target = noise - x_0
    """
    B = x0.shape[0]
    if noise is None:
        noise = torch.randn_like(x0)
    t = torch.rand(B, device=x0.device)
    t_expand = t.view(B, *([1] * (x0.dim() - 1)))
    x_t = (1 - t_expand) * x0 + t_expand * noise
    target_v = noise - x0
    return t, x_t, target_v, noise


@torch.no_grad()
def euler_sample(model, z, num_steps, return_trajectory=False):
    """Euler ODE solver from noise (t=1) to data (t=0).

    Args:
        model: velocity field v(x, t)
        z: initial noise at t=1, shape (B,C,H,W)
        num_steps: number of Euler steps
        return_trajectory: if True, return all intermediate states
    Returns:
        x_0: generated samples
    """
    dt = 1.0 / num_steps
    x = z.clone()
    trajectory = [x] if return_trajectory else None

    for i in range(num_steps):
        t_val = 1.0 - i * dt
        t = torch.full((z.shape[0],), t_val, device=z.device)
        v = model(x, t)
        x = x - v * dt  # Move from t toward 0
        if return_trajectory:
            trajectory.append(x)

    if return_trajectory:
        return x, trajectory
    return x


@torch.no_grad()
def teacher_solve(model, x_t, t_start, num_steps, return_checkpoints=None):
    """Run teacher ODE from t_start to t=0 with given number of steps.

    Used to generate distillation targets.

    Args:
        model: teacher velocity field
        x_t: state at time t_start, shape (B,C,H,W)
        t_start: starting time (scalar or batch), values in (0, 1]
        num_steps: number of Euler steps
        return_checkpoints: optional list of step counts at which to return
            intermediate states (e.g. [2, 5, 10] returns state after 2, 5, 10 steps).
            Must be sorted and <= num_steps.
    Returns:
        x_0_hat if return_checkpoints is None, else dict mapping step_count -> state
    """
    if isinstance(t_start, (int, float)):
        t_start = torch.full((x_t.shape[0],), t_start, device=x_t.device)

    x = x_t.clone()
    dt = t_start / num_steps  # (B,)
    dt_expand = dt.view(-1, 1, 1, 1)  # (B, 1, 1, 1)

    checkpoints = {}
    checkpoint_set = set(return_checkpoints) if return_checkpoints else set()

    for i in range(num_steps):
        frac = i / num_steps
        t_curr = t_start * (1 - frac)  # (B,)
        v = model(x, t_curr)
        x = x - v * dt_expand

        if (i + 1) in checkpoint_set:
            checkpoints[i + 1] = x.clone()

    if return_checkpoints is not None:
        checkpoints[num_steps] = x
        return checkpoints

    return x
