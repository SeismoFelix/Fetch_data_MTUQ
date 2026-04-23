"""
Utilities for manipulating MTUQ and Specfem3D files.

Contains functions for dynamically filtering weight files by distance 
and extracting Specfem3D STATIONS files from MTUQ-formatted SAC headers.
"""

import os
import math
import logging
import obspy
from pathlib import Path

# Setup basic logging for the utilities
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def filter_weight_file(input_weight_file, output_directory, distance_ranges):
    """
    Filters an MTUQ weights.dat file based on specified distance ranges.

    This is useful for iteratively testing different distance cutoffs 
    without regenerating the entire dataset.

    Args:
        input_weight_file (str): Path to the original MTUQ weights.dat file.
        output_directory (str): Directory where the filtered files will be saved.
        distance_ranges (list of tuple): A list containing tuples of (min_dist, max_dist)
                                         in kilometers. E.g., [(0, 150), (150, 300)].

    Returns:
        None: Writes new filtered weight files directly to the output directory.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Read the master weight file once
    with open(input_weight_file, 'r') as f:
        lines = f.readlines()
    
    # Loop through each requested distance range
    for d1, d2 in distance_ranges:
        output_file = os.path.join(output_directory, f"weights_{d1}_{d2}.dat")
        
        with open(output_file, 'w') as out_f:
            for line in lines:
                parts = line.split()
                if len(parts) < 2:
                    continue # Skip empty or malformed lines
                
                try:
                    # Column 2 (index 1) contains the distance in km
                    distance = float(parts[1])
                    
                    # If distance falls within the range, write it to the new file
                    if d1 <= distance < d2:
                        out_f.write(line)
                except ValueError:
                    continue # Skip lines where distance is not a valid float
                    
        print(f"Filtered weight file created: {output_file}")


def coords_match(dict1, dict2):
    """
    Helper function to check if coordinates between two station dictionaries are identical.
    
    Args:
        dict1 (dict): First station data dictionary.
        dict2 (dict): Second station data dictionary.
        
    Returns:
        bool: True if coordinates match within a tolerance of 1e-5, False otherwise.
    """
    return (
        math.isclose(dict1['Latitude'], dict2['Latitude'], abs_tol=1e-5) and
        math.isclose(dict1['Longitude'], dict2['Longitude'], abs_tol=1e-5) and
        math.isclose(dict1['Elevation'], dict2['Elevation'], abs_tol=1e-5) and
        math.isclose(dict1['Burial'], dict2['Burial'], abs_tol=1e-5)
    )


def write_stations_file_specfem(sac_directory, output_filename="STATIONS"):
    """
    Reads MTUQ-formatted SAC files and generates a deduplicated Specfem3D STATIONS file.

    It extracts headers (Network, Station, Location, Lat, Lon, Elev, Depth) and 
    actively resolves Location Code conflicts (e.g., prioritizing '00' over '10').

    Args:
        sac_directory (str): Path to the directory containing MTUQ formatted SAC files.
        output_filename (str): Name of the output file. Defaults to "STATIONS".

    Returns:
        None: Writes the STATIONS file directly to the current working directory.
    """
    sac_dir = Path(sac_directory).expanduser().resolve()
    sac_files = list(sac_dir.glob("*.*.*.*.*.*"))
    
    if not sac_files:
        logging.error(f"No MTUQ formatted files found in {sac_directory}.")
        return

    unique_stations = {}

    for sac_file in sac_files:
        try:
            # headonly=True prevents loading the entire waveform data into memory
            st = obspy.read(str(sac_file), headonly=True)
            tr = st[0]
            
            # Extract basic headers
            net = tr.stats.network
            sta = tr.stats.station
            loc = tr.stats.location.strip() 
            
            # Extract geodetics from the SAC specific header
            lat = tr.stats.sac.get('stla', 0.0)
            lon = tr.stats.sac.get('stlo', 0.0)
            elev = tr.stats.sac.get('stel', 0.0)
            burial = tr.stats.sac.get('stdp', 0.0) 
            
            sta_key = f"{net}.{sta}"
            new_data = {'Location': loc, 'Station': sta, 'Network': net, 
                        'Latitude': lat, 'Longitude': lon, 'Elevation': elev, 'Burial': burial}

            # Handle coordinate conflicts if the station already exists in our dictionary
            if sta_key not in unique_stations:
                unique_stations[sta_key] = new_data
            else:
                existing_data = unique_stations[sta_key]
                if existing_data['Location'] != loc and not coords_match(existing_data, new_data):
                    logging.warning(f"Mismatch for {sta_key}: Loc '{existing_data['Location']}' vs '{loc}'.")
                    if loc == '00':
                        unique_stations[sta_key] = new_data # Prioritize location code '00'

        except Exception as e:
            if not sac_file.name.startswith("."):
                logging.error(f"Failed to read headers from {sac_file.name}: {e}")

    # Write the deduplicated dictionary to the output file
    output_path = Path.cwd() / output_filename
    with open(output_path, 'w') as f:
        for sta_data in unique_stations.values():
            line = (f"{sta_data['Station']:<6} {sta_data['Network']:<4} "
                    f"{sta_data['Latitude']:>9.4f}  {sta_data['Longitude']:>10.4f}  "
                    f"{sta_data['Elevation']:>6.1f}  {sta_data['Burial']:>5.1f}\n")
            f.write(line)
            
    logging.info(f"Success! {output_filename} created with {len(unique_stations)} unique stations.")