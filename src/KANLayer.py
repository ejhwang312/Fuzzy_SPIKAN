# Copyright (c) 2024 Spyros Rigas, Michalis Papachristou
# Adapted from: https://github.com/srigas/jaxKAN
import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.linen import initializers

from splines import get_spline_basis
from general import solve_full_lstsq


class KANLayer(nn.Module):
    """
    KANLayer class.

    Args:
    -----------
        n_in (int): number of layer's incoming nodes. Default: 2
        n_out (int): number of layer's outgoing nodes. Default: 5
        k (int): the order of the spline basis functions. Default: 3
        const_spl (float/bool): coefficient of the spline function in the overall activation. If set to False, then it is trainable per activation. Default: False
        const_res (float/bool): coefficient of the residual function in the overall activation. If set to False, then it is trainable per activation. Default: False
        residual (nn.Module): function that is applied on samples to calculate residual activation. Default: nn.swish
        noise_std (float): noise for the initialization of spline coefficients. Default: 0.1
        grid_e (float): parameter that defines if the grids are uniform (grid_e = 1.0) or sample-dependent (grid_e = 0.0). Intermediate values correspond to a linear mixing of the two cases. Default: 0.05
    """
    n_in: int = 2
    n_out: int = 5
    k: int = 3
    mode: str = 'spikan'  # 'spikan', 'or-spikan', 'xor-spikan' 선택 가능

    const_spl: float or bool = False
    const_res: float or bool = False
    residual: nn.Module = nn.swish

    noise_std: float = 0.1
    grid_e: float = 0.15

    def setup(self):
        """
        Registers and initializes all parameters of the KANLayer. Specifically:
            * grid: non-trainable variable
            * c_basis, c_spl, c_res: trainable parameters
        """
        init_G = 3
        init_knot = (-1, 1)

        # Calculate the step size for the knot vector based on its end values
        h = (init_knot[1] - init_knot[0]) / init_G

        # Create the initial knot vector and perform augmentation
        # Now it is expanded from G+1 points to G+1 + 2k points, because k points are appended at each of its ends
        grid = jnp.arange(-self.k, init_G + self.k + 1, dtype=jnp.float32) * h + init_knot[0]

        # Expand for broadcasting - the shape becomes (n_in*n_out, G + 2k + 1), so that the grid
        # can be passed in all n_in*n_out spline basis functions simultaneously
        grid = jnp.expand_dims(grid, axis=0)
        grid = jnp.tile(grid, (self.n_in * self.n_out, 1))

        # Store the grid as a non trainable variable
        self.grid = self.variable('state', 'grid', lambda: grid)

        # Register & initialize the spline basis functions' coefficients as trainable parameters
        # They are drawn from a normal distribution with zero mean and an std of noise_std
        self.c_basis = self.param('c_basis', initializers.normal(stddev=self.noise_std),
                                  (self.n_in * self.n_out, self.grid.value.shape[1] - 1 - self.k))

        # If const_spl is set as a float value, treat it as non trainable and pass it to the c_spl array with shape (n_in*n_out)
        # Otherwise register it as a trainable parameter of the same size and initialize it
        if isinstance(self.const_spl, float):
            self.c_spl = jnp.ones(self.n_in * self.n_out) * self.const_spl
        elif self.const_spl is False:
            self.c_spl = self.param('c_spl', initializers.constant(1.0), (self.n_in * self.n_out,))

        if isinstance(self.const_res, float):
            self.c_res = jnp.ones(self.n_in * self.n_out) * self.const_res
        elif self.const_res is False:
            self.c_res = self.param('c_res', initializers.constant(1.0), (self.n_in * self.n_out,))

    def get_g(self):
        return self.grid.value

    def basis(self, x):
        """
        Uses k and the current grid to calculate the values of spline basis functions on the input.

        Args:
        -----
            x (jnp.array): inputs
                shape (batch, n_in)

        Returns:
        --------
            bases (jnp.array): spline basis functions applied on inputs
                shape (n_in*n_out, G+k, batch)
        """
        batch = x.shape[0]
        x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(self.n_out, )).reshape((batch, self.n_in * self.n_out))
        x_ext = jnp.transpose(x_ext, (1, 0))

        grid = self.grid.value
        bases = get_spline_basis(x_ext, grid, self.k)
        return bases

    def new_coeffs(self, x, ciBi):
        """
        Utilizes the new grid's basis functions, Bj(x), and the old grid's splines, Sum(ciBix(x)), to approximate a good initialization for the updated grid's parameters.

        Args:
        -----
            x (jnp.array): inputs
                shape (batch, n_in)
            ciBi (jnp.array): values of old grid's splines calculated on the inputs
                shape (n_in*n_out, batch)

        Returns:
        --------
            cj (jnp.array): new coefficients corresponding to the updated grid
                shape (n_in*n_out, G'+k)
        """
        A = self.basis(x)
        Bj = jnp.transpose(A, (0, 2, 1))
        ciBi = jnp.expand_dims(ciBi, axis=-1)
        cj = solve_full_lstsq(Bj, ciBi)
        return jnp.squeeze(cj, axis=-1)

    def update_grid(self, x, G_new):
        """
        Performs a grid update given a new value for G, G_new, and adapts it to the given data, x.

        Args:
        -----
            x (jnp.array): inputs
                shape (batch, n_in)
            G_new (int): Size of the new grid (in terms of intervals)

        Returns:
        --------
            cj (jnp.array): spline function coefficients
                shape (n_in*n_out, G_new+k)
        """
        Bi = self.basis(x)
        ci = self.c_basis
        ciBi = jnp.einsum('ij,ijk->ik', ci, Bi)

        batch = x.shape[0]
        x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(self.n_out, )).reshape((batch, self.n_in * self.n_out))
        x_ext = jnp.transpose(x_ext, (1, 0))
        x_sorted = jnp.sort(x_ext, axis=1)

        ids = jnp.concatenate((jnp.floor(batch / G_new * jnp.arange(G_new)).astype(int), jnp.array([-1])))
        grid_adaptive = x_sorted[:, ids]

        margin = 0.01
        uniform_step = (x_sorted[:, -1] - x_sorted[:, 0] + 2 * margin) / G_new
        grid_uniform = (jnp.arange(G_new + 1, dtype=jnp.float32) * uniform_step[:, None]
                        + x_sorted[:, 0][:, None] - margin)

        grid = self.grid_e * grid_uniform + (1.0 - self.grid_e) * grid_adaptive
        h = (grid[:, [-1]] - grid[:, [0]]) / G_new
        left = jnp.squeeze((jnp.arange(self.k, 0, -1) * h[:, None]), axis=1)
        right = jnp.squeeze((jnp.arange(1, self.k + 1) * h[:, None]), axis=1)
        grid = jnp.concatenate([grid[:, [0]] - left, grid, grid[:, [-1]] + right], axis=1)

        self.grid.value = grid
        return self.new_coeffs(x, ciBi)

    def __call__(self, x):
        batch = x.shape[0]
        x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(self.n_out, )).reshape((batch, self.n_in * self.n_out))
        x_ext = jnp.transpose(x_ext, (1, 0))

        res = jnp.transpose(self.residual(x_ext), (1, 0))
        Bi = self.basis(x)
        spl = jnp.transpose(jnp.einsum('ij,ijk->ik', self.c_basis, Bi), (1, 0))

        cnst_spl = jnp.expand_dims(self.c_spl, axis=0)
        cnst_res = jnp.expand_dims(self.c_res, axis=0)
        y = (cnst_spl * spl) + (cnst_res * res)
        y_reshaped = jnp.reshape(y, (batch, self.n_out, self.n_in))

        # Aggregation mode
        if self.mode == 'spikan':
            # sum
            y = (1.0 / self.n_in) * jnp.sum(y_reshaped, axis=2)
        elif self.mode == 'or-spikan':
            # Fuzzy Algebraic Sum
            y = 1.0 - jnp.prod(1.0 - y_reshaped, axis=2)
        elif self.mode == 'xor-spikan':
            # Generalized XOR (Parity logic)
            y = 0.5 * (1.0 - jnp.prod(1.0 - 2.0 * y_reshaped, axis=2))

        elif self.mode == 'tanh_or-spikan':
            # Fuzzy Algebraic Sum
            y_squashed = jnp.tanh(y_reshaped) /  jnp.tanh(1.0)
            y = 1.0 - jnp.prod(1.0 - y_squashed, axis=2)

        elif self.mode == 'tanh_xor-spikan':
            # Generalized XOR
            y_squashed = jnp.tanh(y_reshaped) / jnp.tanh(1.0)
            y = 0.5 * (1.0 - jnp.prod(1.0 - 2.0 * y_squashed, axis=2))

        # elif self.mode == 'leaky_relu_or-spikan':
        #     # Fuzzy Algebraic Sum
        #     y_squashed = jax.nn.leaky_relu(y_reshaped)
        #     y = 1.0 - jnp.prod(1.0 - y_squashed, axis=2)
        #
        # elif self.mode == 'leaky_relu_xor-spikan':
        #     # Generalized XOR (Parity logic)
        #     y_squashed = jax.nn.leaky_relu(y_reshaped)
        #     y = 0.5 * (1.0 - jnp.prod(1.0 - 2.0 * y_squashed, axis=2))

        elif self.mode == 'sigmoid_or-spikan':
            # Fuzzy Algebraic Sum
            y_squashed = (jax.nn.sigmoid(y_reshaped) -0.5) * 2*(1+jnp.exp(-1)) / (1-jnp.exp(-1))
            y = 1.0 - jnp.prod(1.0 - y_squashed, axis=2)

        elif self.mode == 'sigmoid_xor-spikan':
            # Generalized XOR
            y_squashed = (jax.nn.sigmoid(y_reshaped) -0.5) * 2*(1+jnp.exp(-1)) / (1-jnp.exp(-1))
            y = 0.5 * (1.0 - jnp.prod(1.0 - 2.0 * y_squashed, axis=2))


        grid_reshaped = self.grid.value.reshape(self.n_out, self.n_in, -1)
        input_norm = grid_reshaped[:, :, -1] - grid_reshaped[:, :, 0] + 1e-5
        spl_reshaped = jnp.reshape(spl, (batch, self.n_out, self.n_in))
        spl_reg = (jnp.mean(jnp.abs(spl_reshaped) ** 2, axis=0)) / input_norm

        return y, spl_reg