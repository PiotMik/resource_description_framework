from enum import Enum
from abc import ABC, abstractmethod


class FreqStr(str, Enum):
    """Represents available frequencies.

    Subclass of a string, restricted to the pre-defined value set.
    Implements methods for comparison of two object of this class (obj_1 smaller/greater than obj_2).
    """
    D = "D"
    M = "M"
    Q = "Q"
    A = "A"

    @property
    def numbervalue(self) -> int:
        """Maps frequency string to a number to allow ordering and comparison."""
        return {'D': 1, 'M': 2, 'Q': 3, 'A': 4}[self.value]

    def __ge__(self, other):
        return self.numbervalue >= other.numbervalue

    def __gt__(self, other):
        return self.numbervalue > other.numbervalue

    def __le__(self, other):
        return self.numbervalue <= other.numbervalue

    def __lt__(self, other):
        return self.numbervalue < other.numbervalue

    @classmethod
    def ALL(cls):
        """Returns all available frequencies."""
        return {cls(f) for f in ['D', 'M', 'Q', 'A']}


class FrequencyHandler(ABC):
    """Abstract class responsible for handling the behaviour of frequencies across calculations.
    Contains a INPUTS_N variable, which denotes the maximal number of input frequencies.
    Enforces the implementation of the output_frequency classmethod, which given inputs' frequencies
    gives the frequency of the output.
    """
    INPUTS_N = None

    @abstractmethod
    def output_frequency(self, *input_freqs):
        if isinstance(input_freqs, str):
            input_freqs = [input_freqs]

        if self.INPUTS_N is not None:
            if len(input_freqs) != self.INPUTS_N:
                raise TypeError(f"Accepts exactly {self.INPUTS_N} inputs.")


class PassthroughFrequency(FrequencyHandler):
    """Represents handling of a single time series, without frequency alteration.

    Frequency handler for calculations which require a single time series,
    and output a single time series of the same frequency.

    Example use case: Modifying time series by a constant.
    """
    INPUTS_N = 1

    def output_frequency(self, input_freq: FreqStr):
        super().output_frequency(input_freq)
        return FreqStr(input_freq)


class IdenticalFrequencies(FrequencyHandler):
    """Represents handling of a multiple, same frequencies.

    Frequency handler for calculations which require multiple time series of
    the same frequency, and return the output of the same frequency.

    Example use case: Addition of multiple time series.
    """
    INPUTS_N = None

    def __init__(self, n: int = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.INPUTS_N = n

    def output_frequency(self, *input_freqs):
        super().output_frequency(*input_freqs)
        if len(set(input_freqs)) > 1:
            raise ValueError(f'Needs identical frequencies. Got: {set(input_freqs)}')
        return FreqStr(input_freqs[0])


class CoarsestFrequency(FrequencyHandler):

    def output_frequency(self, *input_freqs):
        super().output_frequency(*input_freqs)
        return max([FreqStr(fr) for fr in input_freqs])


class FinestFrequency(FrequencyHandler):

    def output_frequency(self, *input_freqs):
        super().output_frequency(*input_freqs)
        return min([FreqStr(fr) for fr in input_freqs])


class DownsampleFrequency(FrequencyHandler):
    """Represents handling of a single time series, with downsampling frequency alteration.

    Frequency handler for calculations which require a single time series,
    and output a single time series of a less granular frequency.

    Example use case: Transform a monthly series to quarterly totals.
    """
    INPUTS_N = 1

    def __init__(self, target_freq, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_freq = FreqStr(target_freq)

    def output_frequency(self, input_freq: FreqStr):
        super().output_frequency(input_freq)
        if FreqStr(input_freq) > self.target_freq:
            raise ValueError(f"Can only aggregate to higher frequency.")
        return self.target_freq
