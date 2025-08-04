import numpy as np


def sample_validator(array: np.array, silence_threshold: float) -> bool:
    """Returns True if silence rate is less than
     or equal to silence_threshold"""

    loud_cells = (np.sum(array, axis=0) != 0).sum()
    silence_rate = 1-(loud_cells/array.shape[1])

    return silence_rate <= silence_threshold


def reducer_sampler(array: np.array,
                    duration: float,
                    sample_seconds: int,
                    silence_threshold: float,
                    attempts: int = 100) -> tuple[int, int, np.array]:
    """Reduces 3d tensor to 2d tensor and samples n random columns."""

    reduced_array = np.sum(array, axis=0)
    ticks_per_second = int(reduced_array.shape[1]/duration)
    n = ticks_per_second * sample_seconds

    passed_validation = False

    while attempts > 0:
        index = np.random.randint(0, reduced_array.shape[1] - n)
        sample = reduced_array[:, index:index + n]

        passed_validation = sample_validator(sample, silence_threshold)

        if passed_validation:
            break
        attempts -= 1

    if passed_validation:
        return index, index+n, sample
    else:
        raise ValueError("Could not pull a valid sample.")
