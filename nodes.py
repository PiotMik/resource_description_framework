from abc import ABC, abstractmethod
import frequencies as freqs
import inputs as inp
import pandas as pd
from functools import reduce, wraps


class Calculation(ABC):
    """A single calculation unit.

    Defines admissible input time series frequencies.
    Validates the input with respect to that frequency.
    Given an input frequency, is able to """
    check_input_order = False
    check_output_order = False
    frequency_handler: freqs.FrequencyHandler
    inputs_handler: inp.CalcInput
    _result_cache = None

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def compute(self, inputs):
        """The main computation logic of this unit."""
        pass

    @property
    def check_inputs(self):
        return self.inputs_handler.check_inputs

    @property
    def output_frequency(self):
        return self.frequency_handler.output_frequency


class Filter(Calculation):
    frequency_handler = freqs.PassthroughFrequency()
    inputs_handler = inp.SingleDataFrameInput()

    def __init__(self, names, collapse=False):
        self.ts_names = names
        self.collapse = collapse

    def compute(self, dataframe) -> [pd.Series, pd.DataFrame]:
        self.check_inputs(dataframe)
        filtered_data = dataframe[list(self.ts_names)]
        if len(filtered_data.columns) == 1 and self.collapse:
            filtered_data = filtered_data.iloc[:, 0]

        self._result_cache = filtered_data.copy()
        return filtered_data.copy()


class Output(Calculation):
    frequency_handler = freqs.PassthroughFrequency()
    inputs_handler = inp.SingleSeriesInput()

    def __init__(self, ts_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(ts_name, str):
            raise TypeError('Output node accepts a single string as the output name.')
        self.ts_name = ts_name

    def __str__(self):
        return f"{super().__str__()}: {self.ts_name}"

    def compute(self, time_series: pd.Series) -> pd.Series:
        self.check_inputs(time_series)
        renamed_series = time_series.copy()
        renamed_series.name = self.ts_name
        self._result_cache = renamed_series.copy()
        return renamed_series.copy()


class Input(Calculation):
    frequency_handler = freqs.PassthroughFrequency()
    inputs_handler = inp.SingleDataFrameInput()

    def __init__(self, ts_name):
        if not isinstance(ts_name, str):
            raise TypeError(f'Input node needs a single input name.')
        self.ts_name = ts_name

    def __str__(self):
        return f"{super().__str__()}({self.ts_name})"

    def compute(self, dataframe: pd.DataFrame) -> pd.Series:
        self.check_inputs(dataframe)
        input_ts = dataframe[self.ts_name]
        self._result_cache = input_ts.copy()
        return input_ts.copy()


class Add(Calculation):
    frequency_handler = freqs.IdenticalFrequencies()
    inputs_handler = inp.MultipleSeriesInputs()

    def __init__(self, const_float=0.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const_float = const_float

    def __str__(self):
        arg_str = f"({self.const_float})" if self.const_float != 0.0 else ""
        return f"{super().__str__()}{arg_str}"

    def compute(self, *ts_n):
        ts_n = list(ts_n)
        output = reduce(lambda x, y: x + y, ts_n) + self.const_float
        self._result_cache = output.copy()
        return output.copy()


class Subtract(Calculation):
    check_input_order = True
    frequency_handler = freqs.IdenticalFrequencies(n=2)
    inputs_handler = inp.NSeriesInput(n=2)

    def __init__(self, const_float=0.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const_float = const_float

    def __str__(self):
        arg_str = f"({self.const_float})" if self.const_float != 0.0 else ""
        return f"{super().__str__()}{arg_str}"

    def compute(self, *inputs) -> pd.Series:
        self.check_inputs(*inputs)
        ts_1, ts_2 = inputs
        output = ts_1 - ts_2 - self.const_float
        self._result_cache = output.copy()
        return output.copy()


class Multiply(Calculation):
    frequency_handler = freqs.IdenticalFrequencies()
    inputs_handler = inp.MultipleSeriesInputs()

    def __init__(self, const_float=1.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const_float = const_float

    def __str__(self):
        arg_str = f"({self.const_float})" if self.const_float != 1.0 else ""
        return f"{super().__str__()}{arg_str}"

    def compute(self, *ts_n):
        super().check_inputs(*ts_n)
        ts_n = list(ts_n)
        output = reduce(lambda x, y: x * y, ts_n) * self.const_float
        self._result_cache = output.copy()
        return output.copy()


class Aggregate(Calculation):
    frequency_handler: freqs.DownsampleFrequency
    inputs_handler = inp.SingleInput()

    def __init__(self, frequency, aggregation):
        self.frequency = freqs.FreqStr(frequency)
        self.aggregation = aggregation
        self.frequency_handler = freqs.DownsampleFrequency(target_freq=frequency)

    def __str__(self):
        return f"Aggregate({self.frequency.value}-{self.aggregation})"

    def compute(self, data: [pd.DataFrame, pd.Series]):
        output = getattr(data.resample(self.frequency), self.aggregation)()
        self._result_cache = output.copy()
        return output.copy()


if __name__ == "__main__":
    pass
