from typing import Union
import numpy as np
from util import util, EasyDict

# Camera distributions
class Distribution:
    """Superclass for all distributions."""
    def __init__(self, sampler_config: EasyDict) -> None:
        self.sampler = util.instantiate(sampler_config)

class Sphere(Distribution):
    """Distribute points uniformly on the unit sphere."""
    def __init__(self, sampler_config: EasyDict=EasyDict({'module': 'data.sampler.Independent', 'd': 2}), u_range: list=[0,1.], v_range: list=[0,1.]) -> None:
        super().__init__(sampler_config)
        self.map_range = np.stack([1 - 2 * np.array(u_range), 2 * np.pi * np.array(v_range)], -1)

    def __call__(self) -> np.array:
        x = self.sampler()
        x_map = (1 - x) * self.map_range[0] + x * self.map_range[1]
        y = np.array([np.cos(x_map[1]) * np.sqrt(1 - x_map[0] ** 2), np.sin(x_map[1]) * np.sqrt(1 - x_map[0] ** 2), x_map[0]])
        return y

def Hemisphere(axis=2, **kwargs):
    """Distribute points uniformly on a hemisphere."""
    u_range = [0, 1.]
    v_range = [0, 1.]
    if axis == 0:
        v_range = [-.25, .25]
    elif axis == 1:
        v_range = [0, .5]
    elif axis == 2:
        u_range = [0, .5]

    return Sphere(u_range=u_range, v_range=v_range, **kwargs)

class AABB(Distribution):
    """Distribute points in an axis aligned bounding box."""
    def __init__(self, sampler_config: EasyDict=EasyDict({'module': 'data.sampler.Independent', 'd': 3}), b_0: Union[float, list]=0., b_1: Union[float, list]=1.) -> None:
        super().__init__(sampler_config)
        self.map_range = np.stack([b_0, b_1])

    def __call__(self) -> np.array:
        x = self.sampler()
        y = (1 - x) * self.map_range[0] + x * self.map_range[1]
        return y

class Constant(Distribution):
    """Just return given constants in order."""
    def __init__(self, constants: list=[[0]]) -> None:
        super().__init__(EasyDict({'module': 'data.sampler.Sampler', 'n': len(constants)}))
        self.constants = np.array(constants)

    def __call__(self) -> np.array:
        x = self.constants[self.sampler.idx % self.sampler.n]
        self.sampler()
        return x

def Range(n: int=128, b_0: Union[float, list]=0., b_1: Union[float, list]=1.):
    """Helper method to go through all all parameters with the same step size."""
    return AABB(EasyDict({'module': 'data.sampler.Grid', 'n': n}), b_0, b_1)

class Concat(Distribution):
    """Concatenation of two distributions."""
    def __init__(self, distribution_config_0: EasyDict, distribution_config_1: EasyDict) -> None:
        self.distribution_0 = util.instantiate(distribution_config_0)
        self.distribution_1 = util.instantiate(distribution_config_1)
        if self.distribution_0.sampler.n == -1 or self.distribution_1.sampler.n == -1:
            max_size = -1
        else:
            max_size = max(self.distribution_0.sampler.n, self.distribution_1.sampler.n)
        super().__init__(EasyDict({'module': 'data.sampler.Sampler', 'n': max_size}))

    def __call__(self) -> np.array:
        self.sampler()
        return np.concatenate([self.distribution_0(), self.distribution_1()])