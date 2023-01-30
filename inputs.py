from typing import Optional, Union, List
import pandas as pd
from abc import ABC


class CalcInput(ABC):
    INPUT_N: Optional[int]
    INPUT_TYPES: List[type]

    def check_inputs(self, *inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        self.check_input_number(inputs)
        self.check_input_types(inputs)

    def check_input_number(self, inputs_list):
        if self.INPUT_N is not None:
            if len(inputs_list) != self.INPUT_N:
                raise ValueError(f"Number of inputs expected is: {self.INPUT_N}.")

    def check_input_types(self, inputs):
        if len(self.INPUT_TYPES) == 1:
            types = self.INPUT_TYPES * len(inputs)
        else:
            types = self.INPUT_TYPES

        for input, expected_type in zip(inputs, types):
            if not isinstance(input, expected_type):
                raise TypeError(f"Input is not of type {expected_type}: {input}.")


class SingleInput(CalcInput):
    INPUT_N = 1
    INPUT_TYPES = [type]


class SingleSeriesInput(CalcInput):
    INPUT_N = 1
    INPUT_TYPES = [pd.Series]


class SingleDataFrameInput(CalcInput):
    INPUT_N = 1
    INPUT_TYPES = [pd.DataFrame]


class NSeriesInput(CalcInput):
    INPUT_TYPES = [pd.Series]

    def __init__(self, n):
        super().__init__()
        self.INPUT_N = int(n)


class MultipleSeriesInputs(CalcInput):
    INPUT_N = None
    INPUT_TYPES = [pd.Series]
