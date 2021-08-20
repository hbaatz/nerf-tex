from typing import Union
from math import ceil
import numpy as np
from util import util, EasyDict

# Point samplers
class Sampler:
    """Superclass for all samplers."""
    def __init__(self, d: int=1, n: int=-1, idx: int=0) -> None:
        self.d = d
        self.n = n
        self.idx = idx

    def __call__(self) -> np.array:
        self.idx += 1

    def done(self) -> bool:
        if self.n < 0:
            return False
        else:
            return self.idx >= self.n

class Independent(Sampler):
    """Sample iid at random."""
    def __call__(self) -> np.array:
        super().__call__()
        return np.random.rand(self.d)

class Constant(Sampler):
    """Return given constant."""
    def __init__(self, d: int=1, n: int=0, c: Union[float, list]=0., idx: int=0) -> None:
        super().__init__(d, n, idx)
        if isinstance(c, float):
            c = [c] * d
        self.c = np.array(c, dtype=float)

    def __call__(self) -> np.array:
        super().__call__()
        return self.c

class Grid(Sampler):
    """Sample on a linearly spaced grid."""
    def __init__(self, d: int=1, n: int=-1, idx: int=0, sample_center: bool=False) -> None:
        super().__init__(d, n, idx)
        self.cells_per_d = ceil(self.n ** (1 / self.d))
        self.cell_size = 1 / self.cells_per_d
        self.sample_center = sample_center

    def __call__(self) -> np.array:
        x = np.empty(self.d)
        for i in range(self.d):
            x[i] = (self.idx // (self.cells_per_d ** i)) % self.cells_per_d
  
        x /= self.cells_per_d
        if self.sample_center:
            x += self.cell_size / 2

        super().__call__()

        return x

class Stratified(Grid):
    """Sample by jittering samples in grid cells."""
    def __call__(self) -> np.array:
        return super().sample() + np.random.rand(self.d) * self.cell_size

class Concat(Sampler):
    """Concatenation of two samplers."""
    def __init__(self, sampler_config_0: EasyDict, sampler_config_1: EasyDict, n: int=-1, idx: int=0) -> None:
        sampler_config_0.update({'n': n, 'idx': idx})
        self.sampler_0 = util.instantiate(sampler_config_0)
        sampler_config_1.update({'n': n, 'idx': idx})
        self.sampler_1 = util.instantiate(sampler_config_1)
        super().__init__(self.sampler_0.d + self.sampler_1.d, n, idx)

    def __call__(self) -> np.array:
        super().__call__()
        return np.concatenate([self.sampler_0(), self.sampler_1()])