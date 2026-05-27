from .lightweight_coreset import LightweightCoreset
from .random_sampling import RandomSampling
from .egq_coreset import EGQCoreset

ALGORITHMS = {
    "lightweight": LightweightCoreset,
    "random": RandomSampling,
    "egq": EGQCoreset,
}
