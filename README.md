# Fetch_data_MTUQ

A robust Python package for automating the fetching, processing, and visualization of seismic data for Moment Tensor Uncertainty Quantification (MTUQ).

## 1. Download the Package

If this project is hosted on a Git repository, clone it to your local machine:

    git clone https://github.com/SeismoFelix/Fetch_data_MTUQ.git
    cd Fetch_data_MTUQ

*(If you downloaded the project folder directly, simply extract it and open your terminal inside the Fetch_data_MTUQ root folder).*

##  2. Installation

To ensure stability and prevent dependency conflicts, you should install this package inside its own dedicated Conda environment.

**Step 2.1: Create and activate the environment**

    conda create -n fetch_data_env python=3.10 -y
    conda activate fetch_data_env

**Step 2.2: Install core scientific dependencies**
Because this package relies on Earth Science libraries with heavy C-dependencies, install them via conda-forge first. We will also install jupyter here so you can run the examples.

    conda install -c conda-forge obspy pygmt jupyter -y

**Step 2.3: Install the package**
Make sure your terminal is currently in the root directory of the project (the folder containing the pyproject.toml file) and run:

    pip install -e .

*(Note: Using the -e flag installs the package in "editable" mode. This means if you ever edit the underlying .py scripts, the changes will immediately take effect without you having to reinstall the package!)*

## 3. Quick Start & Examples

The easiest way to learn the workflow is to run the provided example Jupyter Notebook, which contains a fully pre-configured data fetching and plotting pipeline.

**Step 3.1: Navigate to the examples directory**

    cd examples

**Step 3.2: Launch Jupyter Notebook**

    jupyter notebook

**Step 3.3: Run the Pipeline**
Click on the example notebook file (e.g., example.ipynb) to open it in your browser. This interactive notebook will walk you step-by-step through:
1. Loading parameters from project_config.json.
2. Fetching FDSN continuous seismic data and StationXML files.
3. Processing raw data into MTUQ-ready SAC formats (demean, detrend, taper, filter, rotate).
4. Generating dynamic weights.dat files based on distance bands.
5. Creating automated QC plots (Record Sections, Amplitude Decay checks, and PyGMT Station Maps).
