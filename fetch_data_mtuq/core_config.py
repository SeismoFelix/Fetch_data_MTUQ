"""
Core configuration module for the fetch_data_mtuq package.

This module handles the loading and validation of the static JSON 
parameter file used for data acquisition and MTUQ processing.
"""

import json
import os

def load_config(config_path="project_config.json"):
    """
    Loads parameters from the master JSON configuration file.

    Args:
        config_path (str): The relative or absolute path to the JSON 
                           configuration file. Defaults to "project_config.json".

    Returns:
        dict: A dictionary containing all the parameters defined in the JSON file.

    Raises:
        FileNotFoundError: If the specified JSON file does not exist at the path.
    """
    # Verify the file actually exists before trying to open it
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    
    # Open and parse the JSON file into a Python dictionary
    with open(config_path, "r") as f:
        return json.load(f)