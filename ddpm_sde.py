import torch
import numpy as np


class DDPM_SDE:
    def __init__(self, config):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        self.N = config.sde.N
        self.beta_0 = config.sde.beta_min
        self.beta_1 = config.sde.beta_max
        self.ndim = 3 # C, H, W

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        """
        Calculate drift coeff. and diffusion coeff. in forward SDE
        """
        beta = (self.beta_1 - self.beta_0)*t + self.beta_0 # linear beta(t)
        dims = [1] * self.ndim
        beta = beta.view(-1, *dims) # unsqueeze beta to have same num of dims as object
        drift = -beta * x / 2
        diffusion = torch.sqrt(beta)
        return drift, diffusion

    def marginal_prob(self, x_0, t):
        """
        Calculate marginal q(x_t|x_0)'s mean and std
        """
        B = (self.beta_1 - self.beta_0) * t**2/2 + self.beta_0 * t
        dims = [1] * self.ndim
        B = B.view(-1, *dims) # unsqueeze B to have same num of dims as object
        mean = torch.exp(-B/2) * x_0
        std  = torch.sqrt(1-torch.exp(-B))
        return mean, std
    
    def marginal_std(self, t):
        """
        Calculate marginal q(x_t|x_0)'s std
        """
        B = (self.beta_1 - self.beta_0) * t**2/2 + self.beta_0 * t
        dims = [1] * self.ndim
        B = B.view(-1, *dims) # unsqueeze B to have same num of dims as object
        std = torch.sqrt(1-torch.exp(-B))
        return std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def reverse(self, score_fn, ode_sampling=False):
        """Create the reverse-time SDE/ODE.
        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          ode_sampling: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde

        # Build the class for reverse-time SDE.
        class RSDE:
            def __init__(self):
                self.N = N
                self.ode_sampling = ode_sampling

            @property
            def T(self):
                return T

            def sde(self, x, t, y=None):
                """
                Create the drift and diffusion functions for the reverse SDE/ODE.
                
                
                y is here for class-conditional generation through score SDE/ODE
                """
                
                """
                Calculate drift and diffusion for reverse SDE/ODE
                
                
                ode_sampling - True -> reverse SDE
                ode_sampling - False -> reverse ODE
                """
                # forward sde coefficients
                drift, diffusion = sde_fn(x, t)
                if self.ode_sampling:
                    # -forward langevin:
                    drift -= diffusion**2 * score_fn(x, t, y) / 2
                    diffusion = 0
                else:
                    # -2*forward langevin (diffusion doesn't change)
                    drift -= diffusion**2 * score_fn(x, t, y)
                return drift, diffusion
        return RSDE()


class EulerDiffEqSolver:
    def __init__(self, sde, score_fn, ode_sampling = False):
        self.sde = sde
        self.score_fn = score_fn
        self.ode_sampling = ode_sampling
        self.rsde = sde.reverse(score_fn, ode_sampling)

    def step(self, x, t, y=None, backward=True):
        """
        Implement reverse SDE/ODE Euler solver
        """
        

        """
        x_mean = deterministic part
        x = x_mean + noise (yet another noise sampling)
        backward: whether we're solving ODE backward in time, or forward
        """
        dt = self.rsde.T / self.rsde.N
        dw = torch.randn_like(x) * np.sqrt(dt)
        drift, diffusion = self.rsde.sde(x, t, y)
        if backward:
            dt = -dt # minus, as we go backward in time
        x_mean = x + drift * dt
        x = x_mean + diffusion * dw
        return x, x_mean
