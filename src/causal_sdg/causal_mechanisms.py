"""
================================================================================
Author: Andrea Iommi
Code Ownership:
    - All Python source code in this file is written solely by the author.
Documentation Notice:
    - All docstrings and inline documentation are written by ChatGPT,
      but thoroughly checked and approved by the author for accuracy.
================================================================================
"""

import ast
import inspect
from abc import abstractmethod
from typing import Callable, List, Union, Tuple, Any, Literal

import numpy as np
from dowhy import gcm
from dowhy.gcm.causal_mechanisms import ConditionalStochasticModel, StochasticModel
from numpy import ndarray, count_nonzero
from sklearn.preprocessing import MultiLabelBinarizer

from .cpt import CPTData, PTData


class CustomMechanism:
    def __init__(self, num_func: Union[Callable[[Any], int], int] = None):

        self.num_skill_distribution = self.valid_func(num_func)
        self.dim_input = None

    @staticmethod
    def valid_func(num_func: Union[Callable[[Any], int], int] = None):
        """
        Determines and validates a strategy for computing the number of elements. The strategy can be either
        a fixed integer or a callable function. If no valid strategy is provided, a default normal distribution
        strategy is applied where the mean is 5 and the standard deviation is 1.

        Parameters:
        -----------
        num_func: Union[Callable[[Any], int], Callable[[], int], int]
            An integer representing a fixed number of elements, or a callable function determining
            the number of elements with zero or one ignored parameter. If not provided, a default strategy is used.
            Callable functions may return 0 to generate an empty list.

        Returns:
        -----------
        Callable
            A function that determines the number of elements based on the selected strategy.

        Raises:
        -----------
        ValueError: If `num_func` is an integer less than or equal to zero.
        ValueError: If `num_func` returns a negative integer.
        """
        # Set the number-of-elements strategy (either a fixed number or a function)
        if isinstance(num_func, int):
            if num_func <= 0:
                raise ValueError("num_func must be a positive integer")
            else:
                num_skill_distribution = num_func
        elif callable(num_func):
            num_skill_distribution = num_func
        else:
            # Default to a normal distribution with mean=5 and std=1 for the number of elements to sample
            num_skill_distribution = lambda _: max(1, round(np.random.normal(loc=5, scale=1)))
        return num_skill_distribution

    @abstractmethod
    def strategy(self, X: Tuple, elements: int) -> List:
        raise NotImplementedError()

    def check_input(self, X: ndarray) -> None:
        """
        Validates the shape of the input array.

        Ensures that the input is a 1d matrix, corresponding to a single
        sample with n parent values.

        Parameters:
        -----------
        X : np.ndarray
            Input array representing a sample of parent variable values.

        Raises:
        -------
        ValueError
            If the input does not have 1d, i.e., not a single sample.
        """
        if X.ndim != 1:
            raise ValueError("Input must have 1d shape but X has ", X.shape)

        if len(X) != self.dim_input:
            raise ValueError(f"Input has {len(X)} elements but expected {self.dim_input} elements."
                             f"Input: {X}")

    def prepare_input(self, X: ndarray) -> Tuple:
        """
        Converts a single-row input ndarray into a tuple of strings for dictionary lookup.

        Parameters:
        -----------
        X : np.ndarray
            A (1, n) array representing a single parent sample.

        Returns:
        --------
        Tuple
            Tuple elements from X, used as a key in the distribution_dict.

        Raises:
        -------
        ValueError:
            If the input does not represent a single sample.
        """
        if isinstance(X, ndarray):
            self.check_input(X=X)
            return tuple(X)

        elif isinstance(X, (str, float, int)):  # Added int for completeness
            # Assuming a single parent, which should have dim_input of 1
            if self.dim_input != 1:
                raise ValueError(f"Received a single value but expected {self.dim_input} inputs.")
            return (X,)
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

    def get_num_elements(self) -> int:
        """
        Determines the number of elements to sample based on the current strategy.

        Returns:
        --------
        int
            The number of elements to sample, determined by the strategy (either a fixed value or a function).
        """
        if isinstance(self.num_skill_distribution, int):
            return self.num_skill_distribution

        signature = inspect.signature(self.num_skill_distribution)
        if len(signature.parameters) == 0:
            elements = self.num_skill_distribution()
        else:
            elements = self.num_skill_distribution(None)

        elements = int(elements)
        if elements < 0:
            raise ValueError("num_func must return a non-negative integer")

        return elements

    def mechanisms_func(self, X: Union[str, float, ndarray]) -> str:
        """
        This method uses the `strategy` method to sample one or more outcomes based on the
        input key `X`. The number of elements sampled is determined by the `get_num_elements`.

        Parameters:
        -----------
        X : Union[str, float, ndarray]
            The input data representing the parent configuration

        Returns:
        --------
        str
            A string representation of the sampled outcomes from the strategy.

        Notes:
        ------
        - The `prepare_input` method is used to preprocess the input data before passing it to the
          `strategy`.
        - This method is designed to handle both single and multiple sample inputs, making it adaptable
          to various types of input configurations.
        """
        X = self.prepare_input(X=X)
        return str(self.strategy(X=X, elements=self.get_num_elements()))



class StochasticMechanismFromDict(CustomMechanism, StochasticModel):
    """

    """
    def __init__(self, pt_data: PTData, num_func: Union[Callable[[Any], int], int] = 1,
                 return_always_list: bool = False):
        super().__init__(num_func=num_func)

        self.pt_data = pt_data
        self.return_always_list = return_always_list

    def draw_samples(self, num_samples: int) -> np.ndarray:
        """
        Samples from the causal mechanism for a given number of samples.

        Parameters:
        -----------
         num_samples: ndarray
            Number of samples to generate from the mechanism.

        Returns:
        --------
        ndarray
            Array of shape (m,1) or (m, k) containing the generated samples for each
            parent configuration. The output shape depends on what `mechanisms_func` returns.
        """
        # Apply the custom mechanism row-wise to the parent_samples array
        return np.asarray([self.mechanisms_func(np.array([])) for _ in range(num_samples)])

    def strategy(self, X: Tuple, elements: int) -> List:
        """
        Samples one or more output values from the predefined categorical distribution.
        This method retrieves the values and associated probabilities from the predefined distribution
        and samples the specified number of elements (`elements`) based on the probabilities.

        Parameters:
        -----------
        X : Tuple
            A tuple representing the parent configuration, where each element in the tuple is a string
            representing a category or state of the parent variables.

        elements : int
            The number of elements to sample from the categorical distribution for the given key `X`.
            If the number of elements requested exceeds the number of possible non-zero probability elements,
            the method will sample only from the available non-zero probability elements.

        Returns:
        --------
        Union[List[str, float], List]
            A list of sampled outcomes from the distribution. The type of the outcomes can be either:
            - A list of strings, if the outcomes are categorical.
            - A list of floats, if the outcomes are numerical.
            If no elements are sampled (e.g., no valid distribution for `X`), an empty list is returned.

        Example:
        --------
        cpt_data = CPTData(
            cpt={
                ('Full-Time',): np.array([0.1, 0.9]), # P(Y | X='Full-Time')
                ('Part-Time',): np.array([0.8, 0.2]), # P(Y | X='Part-Time')
            },
            y_categories=['Permanent', 'Fixed-Term']
        )
        cpt_mechanism = ConditionalMechanismFromDict(cpt_data=cpt_data)
        result = cpt_mechanism.strategy(X=('Full-Time',), elements=1)
        print(result) -> 'Fixed-Term'

        result = cpt_mechanism.strategy(X=('Part-Time',), elements=2)
        print(sorted(result)) # Sorted for consistent output -> ['Fixed-Term', 'Permanent']

        Raises:
        -------
        KeyError:
            If the input key `X` is not found in the distribution dictionary.
        """
        values, dist = self.pt_data.y_categories, self.pt_data.get_distribution(X=X)
        elements = min(elements, count_nonzero(dist))
        if elements == 0:
            return []

        # Sample the elements based on the distribution probabilities
        sampled_elements = np.random.choice(values, size=elements, replace=False, p=dist).tolist()
        sampled_elements = sampled_elements[0] \
            if not self.return_always_list and elements == 1 else sampled_elements
        return sampled_elements

    def fit(self, X: np.ndarray) -> None:
        """
        No fitting is required since the mechanism is defined by an explicit dictionary.

        Parameters:
        -----------
        X : np.ndarray
            Parent samples (ignored).
        """
        self.dim_input = 0
        return

    def clone(self):
        return StochasticMechanismFromDict(pt_data=self.pt_data.copy(),
                                           num_func=self.num_skill_distribution,
                                           return_always_list=self.return_always_list)


class ConditionalMechanismFromDict(CustomMechanism, ConditionalStochasticModel):
    """
    Implements a custom causal mechanism where the output is sampled from a predefined
    categorical distribution conditioned on discrete input categories (as tuples of strings).

    This class is suitable for modeling conditional probability tables (CPTs) in structural
    causal models (SCMs) where child variables are determined probabilistically given
    categorical parent variables. It also supports an optional number of elements to sample
    for each set of parent values, determined either by a provided function or a fixed value.

    """

    def __init__(self, cpt_data: CPTData, num_func: Union[Callable[[Any], int], int] = None,
                 return_always_list: bool = False, fill_missing: Literal["zero", "uniform"] = "zero",
                 verbose: bool = False):

        """
        Initializes the mechanism with a predefined probability distribution and a strategy for determining
        the number of elements to sample for each parent configuration.

        Parameters:
        -----------
        cpt_data : CPTData
            - keys are tuples of strings representing the parent configuration (e.g., ('a',)),
            - values are probabilities array of probabilities associated with each outcome
            (e.g., np.array([0.4, 0.6])).

        num_func : Union[Callable[[], int], Callable[[Any], int], int], optional
            A function or an integer used to determine the number of elements to sample for each parent configuration.
            - If provided as an integer, it represents a fixed number of elements to sample.
            - If provided as a callable function, it must return a non-negative integer.
              Callables with zero or one parameter are supported.
            - If not provided, defaults to a function that samples from a normal distribution with mean 5 and std 1.

        return_always_list : bool
        If True, sampled outputs are always returned as a list, even if only one element
        is sampled. If False, a single value is returned when one element is sampled.

        fill_missing : Literal["zero", "uniform"]
            Strategy used when a parent configuration `X` is not found in the CPT:
                - "zero": assigns zero-probability to all outcomes.
                - "uniform": assigns equal probability to all outcomes.

        verbose : bool
            If True, provides detailed output when missing configurations are encountered during sampling.

        Example:
        --------
        cpt_data = CPTData(
            cpt={
                ("a",): np.array([0.4, 0.6]),
                ("b",): np.array([0, 1.0])
            },
            y_categories=["horse", "cat"]
        )

        num_func = lambda: np.random.randint(1, 6) # Sample between 1 and 5 elements
        """
        super().__init__(num_func=num_func)

        if not cpt_data.is_valid:
            raise ValueError(f"cpt_data is not valid:")
        if fill_missing not in ("zero", "uniform"):
            raise ValueError("fill_missing must be either 'zero' or 'uniform'.")

        self.cpt_data = cpt_data
        self.return_always_list = return_always_list

        self.fill_missing = fill_missing
        self.verbose = verbose

    def fit(self, X: ndarray, Y: ndarray) -> None:
        """
        No fitting is required since the mechanism is defined by an explicit dictionary.

        Parameters:
        -----------
        X : np.ndarray
            Parent samples (ignored).

        Y : np.ndarray
            Child samples (ignored).
        """
        _, dim_input = X.shape

        if dim_input != self.cpt_data.get_dim_input():
            raise ValueError(f"Input X as {dim_input} elements but expected {self.cpt_data.get_dim_input()} "
                             f"elements in CPT. Input: {X}.")

        self.dim_input = dim_input
        return

    def draw_samples(self, parent_samples: ndarray) -> ndarray:
        """
        Samples from the causal mechanism for each provided set of parent values.

        Parameters:
        -----------
        parent_samples : ndarray
            Array of shape (m, n) where each row represents a configuration of n
            parent variables for which to generate a sample of the child variable.

        Returns:
        --------
        ndarray
            Array of shape (m,1) or (m, k) containing the generated samples for each
            parent configuration. The output shape depends on what `mechanisms_func` returns.
        """
        # Apply the custom mechanism row-wise to the parent_samples array
        return np.asarray([self.mechanisms_func(i) for i in parent_samples])

    def strategy(self, X: Tuple, elements: int) -> List:
        """
        Samples one or more output values from the conditional distribution based on the input key.

        This method retrieves the values and associated probabilities from the predefined distribution
        for the given input key `X` (representing the parent configuration). It then samples the specified
        number of elements (`elements`) based on the probabilities. If the distribution does not contain
        the input key, it returns an empty list. It also ensures that no more elements than those available
        with non-zero probability are sampled.

        Parameters:
        -----------
        X : Tuple
            A tuple representing the parent configuration, where each element in the tuple is a string
            representing a category or state of the parent variables.

        elements : int
            The number of elements to sample from the categorical distribution for the given key `X`.
            If the number of elements requested exceeds the number of possible non-zero probability elements,
            the method will sample only from the available non-zero probability elements.

        Returns:
        --------
        Union[List[str, float], List]
            A list of sampled outcomes from the distribution. The type of the outcomes can be either:
            - A list of strings, if the outcomes are categorical.
            - A list of floats, if the outcomes are numerical.
            If no elements are sampled (e.g., no valid distribution for `X`), an empty list is returned.

        Raises:
        -------
        KeyError:
            If the input key `X` is not found in the distribution dictionary.

        Notes:
        ------
        - The method ensures that the number of elements sampled does not exceed the number of available elements
          with non-zero probability. This is done by using the `count_nonzero` helper function to check the valid
          number of elements to sample.
        - The `np.random.choice` function is used for sampling, with `replace=False` ensuring that there is no
          repetition of elements when the sampling size is less than the number of available options.
        - Return type respects `return_always_list`.

        """
        # Retrieve the list of elements and the associated distribution for the given key.
        values, dist = self.cpt_data.y_categories, self.cpt_data.get_distribution(X=X,
                                                                                  fill_missing=self.fill_missing,
                                                                                  verbose=self.verbose)

        # Ensure that we do not sample more elements than available with non-zero probability.
        elements = min(elements, count_nonzero(dist))  # Ensure valid sampling size
        if elements == 0:
            return []  # Return an empty list if no valid elements to sample
        else:
            # Sample the elements based on the distribution probabilities
            sampled_elements = np.random.choice(values, size=elements, replace=False, p=dist).tolist()
            sampled_elements = sampled_elements[
                0] if not self.return_always_list and elements == 1 else sampled_elements
            return sampled_elements

    def clone(self):
        return ConditionalMechanismFromDict(cpt_data=self.cpt_data.copy(),
                                            num_func=self.num_skill_distribution,
                                            return_always_list=self.return_always_list,
                                            fill_missing=self.fill_missing,
                                            verbose=self.verbose)


class List2NominalMechanism(CustomMechanism, ConditionalStochasticModel):
    """
    A mechanism that models the conditional distribution P(Y | X) where `X` is a list of
    features and `Y` is a nominal class label.

    Internally, it uses a Multinomial Naive Bayes classifier on the binarized representation
    of the input lists (via `MultiLabelBinarizer`).

    Attributes:
    -----------
        X_list_encoder : MultiLabelBinarizer
            Encodes input list features into binary indicator vectors.

        Y_encoder : Dict[str, int]
            Maps each Y category to a unique integer label.

        cls : MultinomialNB
            A multinomial Naive Bayes classifier used to estimate P(Y | X).

    """

    def __init__(self, num_func: Union[Callable[[Any], int], int] = 1, return_always_list: bool = False):
        """

        Initializes the List2NominalMechanism.

        Parameters:
        -----------
        num_func : Union[Callable[[], int], Callable[[Any], int], int], optional
            A function or an integer used to determine the number of elements to sample for each parent configuration.
            - If provided as an integer, it represents a fixed number of elements to sample.
            - If provided as a callable function, it must return a non-negative integer.
              Callables with zero or one parameter are supported.
            - If not provided, defaults to a function that samples from a normal distribution with mean 5 and std 1.

        return_always_list : bool
        If True, sampled outputs are always returned as a list, even if only one element
        is sampled. If False, a single value is returned when one element is sampled.
        """
        super().__init__(num_func)
        self.return_always_list = return_always_list
        self.X_list_encoder = MultiLabelBinarizer()

        self.cls = gcm.ClassifierFCM()

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fits the Naive Bayes model on training data where each input is a list of elements.

        Parameters:
        -----------
        X : np.ndarray
            Array of list-encoded parent values. Each element should be a list-like string or
            actual list, e.g., ["element1", "element2"].

        Y : np.ndarray
            Array of target class labels.
        """
        if X.shape[1] != 1:
            raise ValueError(f"Input X as {X.shape[1]} elements but expected 1 elements. Input: {X}.")

        X, Y = X.reshape(-1), Y.reshape(-1)
        X = [self._parse_list_like(x) for x in X]
        # Binarize input lists (multi-hot encoding)
        X_encoded = self.X_list_encoder.fit_transform(X)

        # Fit the classifier with integer-encoded Y values
        self.cls.fit(X_encoded, Y)

        self.dim_input = 1

    def draw_samples(self, parent_samples: ndarray) -> ndarray:
        """
        Samples from the causal mechanism for each provided set of parent values.

        Parameters:
        -----------
        parent_samples : ndarray
            Array of shape (m, n) where each row represents a configuration of n
            parent variables for which to generate a sample of the child variable.

        Returns:
        --------
        ndarray
            Array of shape (m,1) or (m, k) containing the generated samples for each
            parent configuration. The output shape depends on what `mechanisms_func` returns.
        """
        # Apply the custom mechanism row-wise to the parent_samples array
        return np.asarray([self.mechanisms_func(i) for i in parent_samples])

    def strategy(self, X: Tuple, elements: int) -> List:
        """
        Predicts outcomes from the learned distribution P(Y | X) for a given input list.

        Parameters:
        -----------
        X : Union[str, float, np.ndarray]
            Input in serialized format (a list seen as a str) or array format with shape (1,).

        Example:
        --------

        mechanism = List2NominalMechanism(num_func=1)
        X_train_str = np.array([
            ["['Python', 'SQL']"],
            ["['Java', 'C++']"],
            ["['Python', 'Java']"]
        ])
        Y_train = np.array(['Data Scientist', 'Software Engineer', 'Data Engineer'])
        mechanism.fit(X_train_str, Y_train)

        input_skills = "['Python', 'SQL', 'Spark']"
        result = mechanism.strategy(X=(input_skills,), elements=1)

        # The model learned that 'Python' and 'SQL' are associated with 'Data Scientist'
        print(result) -> 'Data Scientist'


        Returns:
        --------
        str
            Sampled value(s) from the predicted class distribution.
        """

        input_list = self._parse_list_like(X[0])

        # binary encoding
        X_encoded = self.X_list_encoder.transform([input_list])

        # Compute class probabilities and sample outcomes
        sampled_elements = [
            self._unwrap_singleton_nominal(self.cls.draw_samples(X_encoded).flatten()[0])
            for _ in range(elements)
        ]
        return sampled_elements[0] if not self.return_always_list and elements == 1 else sampled_elements

    def clone(self):
        return List2NominalMechanism(num_func=self.num_skill_distribution,
                                     return_always_list=self.return_always_list)

    @staticmethod
    def _unwrap_singleton_nominal(value: Any) -> Any:
        if isinstance(value, ndarray):
            if value.size == 1:
                return value.reshape(-1)[0]
            return value.tolist()
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return value[0]
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value

            if isinstance(parsed, (list, tuple)) and len(parsed) == 1:
                return parsed[0]

        return value

    @staticmethod
    def _parse_list_like(value: Any) -> List:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, ndarray):
            return value.tolist()
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return []

            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, tuple):
                return list(parsed)
            return []

        return []
