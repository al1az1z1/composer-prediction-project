import numpy as np


def sample_validator(array: np.array, silence_threshold: float) -> bool:
    """Returns True if silence rate is less than
     or equal to silence_threshold"""

    loud_cells = (np.sum(array, axis=0) != 0).sum()
    silence_rate = 1-(loud_cells/array.shape[1])

    return silence_rate <= silence_threshold


def reducer_sampler(array: np.array,
                    n: int,
                    silence_threshold: float,
                    attempts: int = 100) -> tuple[int, int, np.array]:
    """Reduces 3d tensor to 2d tensor and samples n random columns."""

    reduced_tensor = np.sum(array, axis=0)

    passed_validation = False
    while passed_validation is not True and attempts > 0:
        index = np.random.randint(0, reduced_tensor.shape[1]-n)
        sample = reduced_tensor[:, index:index+n]
        passed_validation = sample_validator(sample, silence_threshold)
        attempts -= 1

    return index, index+n, sample
