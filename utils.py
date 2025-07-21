import os
import random
import numpy as np
from typing import Optional

def get_shuffled_files(data_dir:str, class_0_name:str, class_1_name:str, seed:Optional[int]=None):
    """
    Loads and shuffles files from two class directories.

    Args:
        data_dir (str): The directory containing the class subdirectories.
        class_0_name (str): The name of the first class subdirectory.
        class_1_name (str): The name of the second class subdirectory.
        seed (int, optional): The random seed for shuffling.

    Returns:
        tuple: A tuple containing two lists of shuffled file paths for each class.
    """
    class_0_files = [os.path.join(data_dir, class_0_name, f) for f in os.listdir(os.path.join(data_dir, class_0_name))]
    class_1_files = [os.path.join(data_dir, class_1_name, f) for f in os.listdir(os.path.join(data_dir, class_1_name))]

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    random.shuffle(class_0_files)
    random.shuffle(class_1_files)

    return class_0_files, class_1_files

def balance_classes(cl1:list, cl2:list):
    """
    Balances two lists of file paths by truncating the longer list to the length of the shorter list.

    Args:
        cl1 (list): The first list of file paths.
        cl2 (list): The second list of file paths.

    Returns:
        tuple: A tuple containing two balanced lists of file paths.
    """
    min_length = min(len(cl1), len(cl2))
    random.shuffle(cl1)
    random.shuffle(cl2)

    class_1_files = cl1[:min_length]
    class_2_files = cl2[:min_length]

    return class_1_files, class_2_files



class OneHotEncoder:
    """
    A simple one-hot encoder for categorical values.
    """
    def __init__(self, unique_values:list):
        """
        Initializes the OneHotEncoder.

        Args:
            unique_values (list): A list of unique values to be encoded.
        """
        self.unique_values = unique_values
        self.value_to_index = {value: idx for idx, value in enumerate(unique_values)}
        self.num_classes = len(unique_values)
    
    def encode(self, value):
        """
        Encodes a value into a one-hot vector.

        Args:
            value: The value to be encoded.

        Returns:
            np.ndarray: The one-hot encoded vector.
        """
        one_hot = np.zeros(self.num_classes, dtype=int)
        if value in self.value_to_index:
            one_hot[self.value_to_index[value]] = 1
        else:
            raise ValueError(f"{value} is not in the known unique values.")
        return one_hot
    
    def decode(self, one_hot:np.ndarray):
        """
        Decodes a one-hot vector into its original value.

        Args:
            one_hot (np.ndarray): The one-hot encoded vector.

        Returns:
            The original value.
        """
        index = np.argmax(one_hot)
        if index < len(self.unique_values):
            return self.unique_values[index]
        else:
            raise ValueError(f"Invalid one-hot encoding: {one_hot}")
