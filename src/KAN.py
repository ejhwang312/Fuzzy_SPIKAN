# Copyright (c) 2024 Spyros Rigas, Michalis Papachristou
# Adapted from: https://github.com/srigas/jaxKAN
from typing import List
from jax import numpy as jnp

from flax import linen as nn
from flax.linen import initializers
from flax.core import unfreeze

from KANLayer import KANLayer


class KAN(nn.Module):
    """
    KAN class, corresponding to a network of KANLayers.

    Args:
    -----------
        layer_dims (List[int]): defines the network in terms of nodes. E.g. [4,5,1] is a network with 2 layers: one with n_in=4 and n_out=5 and one with n_in=5 and n_out = 1.
        add_bias (bool): boolean that controls wether bias terms are also included during the forward pass or not. Default: True
        k (int): input for KANLayer class - see KANLayer.py
        const_spl (float/bool): input for KANLayer class - see KANLayer.py
        const_res (float/bool): input for KANLayer class - see KANLayer.py
        residual (nn.Module): input for KANLayer class - see KANLayer.py
        noise_std (float): input for KANLayer class - see KANLayer.py
        grid_e (float): input for KANLayer class - see KANLayer.py
        mode (str): Aggregation mode ('spikan', 'or-spikan', 'xor-spikan'). Default: 'spikan'
    """
    layer_dims: List[int]
    add_bias: bool = True

    k: int = 3
    const_spl: float or bool = False
    const_res: float or bool = False
    residual: nn.Module = nn.swish
    noise_std: float = 0.1
    grid_e: float = 0.02
    j: str = '0'
    mode: str = 'spikan'  # mode 인자 추가

    def setup(self):
        """
        Registers and initializes all KANLayers of the architecture.
        Optionally includes a trainable bias for each KANLayer.
        """
        # Initialize KAN layers based on layer_dims
        self.layers = [KANLayer(
            n_in=self.layer_dims[i],
            n_out=self.layer_dims[i + 1],
            k=self.k,
            const_spl=self.const_spl,
            const_res=self.const_res,
            residual=self.residual,
            noise_std=self.noise_std,
            grid_e=self.grid_e,
            mode=self.mode  # KANLayer 생성 시 mode 전달
        )
            for i in range(len(self.layer_dims) - 1)
        ]

        if self.add_bias:
            self.biases = [self.param('bias_' + str(i), initializers.zeros, (dim,)) for i, dim in
                           enumerate(self.layer_dims[1:])]

    def get_grids(self, x, G_new):
        """
        Performs the grid update for each layer of the KAN architecture.

        Args:
        -----
            x (jnp.array): inputs for the first layer
                shape (batch, self.layers[0])
            G_new (int): Size of the new grid (in terms of intervals)

        """
        updated_params = unfreeze(self.scope.variables()['params'])
        updated_state = unfreeze(self.scope.variables()['state'])

        for i, layer in enumerate(self.layers):
            layer_variables = {
                'params': updated_params[f'layers_{i}'],
                'state': updated_state[f'layers_{i}']
            }
            grid = layer.get_g()

        return {'params': updated_params, 'state': updated_state}, grid

    def update_grids(self, x, G_new):
        """
        Performs the grid update for each layer of the KAN architecture.

        Args:
        -----
            x (jnp.array): inputs for the first layer
                shape (batch, self.layers[0])
            G_new (int): Size of the new grid (in terms of intervals)

        """
        updated_params = unfreeze(self.scope.variables()['params'])
        updated_state = unfreeze(self.scope.variables()['state'])

        for i, layer in enumerate(self.layers):
            layer_variables = {
                'params': updated_params[f'layers_{i}'],
                'state': updated_state[f'layers_{i}']
            }

            coeffs, updated_layer_state = layer.apply(layer_variables, x, G_new, method=layer.update_grid,
                                                      mutable=['state'])

            updated_params[f'layers_{i}']['c_basis'] = coeffs
            updated_state[f'layers_{i}'] = updated_layer_state['state']

            layer_variables = {
                'params': updated_params[f'layers_{i}'],
                'state': updated_state[f'layers_{i}']
            }
            layer_output, _ = layer.apply(layer_variables, x)
            if self.add_bias:
                layer_output += self.biases[i]
            x = layer_output

        return {'params': updated_params, 'state': updated_state}

    def __call__(self, x):
        """
        Equivalent to the network's forward pass. [cite: 198, 199]
        """
        spl_regs = []

        for i, layer in enumerate(self.layers):
            x, spl_reg = layer(x)

            if self.add_bias:
                x += self.biases[i]

            spl_regs.append(spl_reg)

        return x, spl_regs