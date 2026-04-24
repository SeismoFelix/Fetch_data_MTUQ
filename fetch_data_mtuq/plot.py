"""
Seismic data visualization module.

Provides subroutines to plot filtered 3-component seismograms 
overlaid with theoretical phase arrivals calculated via TauP.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from obspy import read, UTCDateTime
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
import pygmt

def plot_station_data(station_id, data_dir, origin_time_str, freqmin, freqmax, t1, t2, phases, output_file, vel_model="ak135"):
    """
    Plots Z, R, T seismograms for a single station, normalizes them globally, 
    and overlays TauP theoretical phase arrivals.

    Args:
        station_id (str): The station identifier (e.g., "II_UOSS_00").
        data_dir (str): Directory containing the processed SAC files (Z, R, T).
        origin_time_str (str): Earthquake origin time in ISO format (e.g., "2023-08-18T16:10:03").
        freqmin (float): Minimum frequency for the bandpass filter in Hz.
        freqmax (float): Maximum frequency for the bandpass filter in Hz.
        t1 (float): Start of the trim window (seconds relative to trace onset).
        t2 (float): End of the trim window (seconds relative to trace onset).
        phases (list of str): List of seismic phases to calculate via TauP (e.g., ["P", "S"]).
        output_file (str): Path and filename for the output PDF.
        vel_model (str): Name of the velocity model to use for TauP calculations (default: "ak135").

    Returns:
        None: Saves the plot as a PDF to the specified output_file.
    """
    print(f"Preparing plot for {station_id}...")

    # Read the 3 components (Z, R, T) utilizing glob-style wildcards
    try:
        st = read(os.path.join(data_dir, f"*{station_id}*.z"))
        st += read(os.path.join(data_dir, f"*{station_id}*.r"))
        st += read(os.path.join(data_dir, f"*{station_id}*.t"))
        
    except Exception as e:
        print(f"Error: Could not read files for {station_id}: {e}")
        return

    if len(st) != 3:
        print(f"Error: Found {len(st)} components. Expected exactly 3 (Z, R, T).")
        return
    
    # Remove mean to center on zero, remove linear drift, and apply a 5% Hann taper
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.05, type="hann")

    # Apply zerophase bandpass filter
    st.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)

    # Trim the data relative to the start of the recorded trace
    trace_onset = st[0].stats.starttime
    trim_start = trace_onset + t1
    trim_end = trace_onset + t2
    st.trim(trim_start, trim_end)

    if len(st[0].data) == 0:
        print(f"Error: Trim window {t1}s to {t2}s is outside the bounds of the data.")
        return

    # Normalize waveforms to the largest amplitude across ALL 3 components to preserve relative scale
    global_max_amp = max([np.max(np.abs(tr.data)) for tr in st])
    for tr in st:
        tr.data = tr.data / global_max_amp

    # Extract metadata for TauP theoretical arrival calculation
    header = st[0].stats.sac
    evdp_km = header.evdp
    dist_deg = locations2degrees(header.evla, header.evlo, header.stla, header.stlo)

    # Calculate phase arrivals using the specified 1D velocity model
    model = TauPyModel(model=vel_model)
    print(f"Calculating TauP arrivals for {station_id}:")
    print(f"  Event Depth: {evdp_km} km, Distance: {dist_deg:.2f} degrees")
    arrivals = model.get_travel_times(source_depth_in_km=evdp_km, 
                                      distance_in_degree=dist_deg, 
                                      phase_list=phases)
    print(arrivals)
    # Setup the Matplotlib figure
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    plt.subplots_adjust(hspace=0) 
    components = ['Z', 'R', 'T']
    
    # Plot Seismograms
    for ax, comp in zip(axes, components):
        tr = st.select(component=comp)[0]
        times = tr.times() + t1 
        
        ax.plot(times, tr.data, color='black', linewidth=1.0, label=f"{station_id} - {comp}")
        ax.set_ylim(-1.2, 1.2) 
        ax.set_ylabel("Norm. Amp")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)

    # Plot Phase Arrivals
    origin_time = UTCDateTime(origin_time_str)
    plotted_phases = set() 
    
    for arrival in arrivals:
        if arrival.name in plotted_phases:
            continue
            
        plotted_phases.add(arrival.name)
        
        # Convert absolute arrival time to relative time matching the x-axis
        absolute_arrival_time = origin_time + arrival.time
        relative_arrival_time = absolute_arrival_time - trace_onset

        # Only plot the arrival if it falls within our trim window
        if t1 <= relative_arrival_time <= t2:
            phase_color = 'red' if arrival.name.upper().startswith('P') else 'blue' if arrival.name.upper().startswith('S') else 'purple'
            
            for ax in axes:
                ax.axvline(x=relative_arrival_time, color=phase_color, linestyle='--', linewidth=0.5)
            
            axes[0].text(relative_arrival_time, 1.02, arrival.name, color=phase_color, 
                         ha='center', va='bottom', transform=axes[0].get_xaxis_transform(), 
                         fontsize=10, fontweight='light')

    axes[-1].set_xlabel("Time (seconds) relative to trace onset")
    
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"  ✓ Plot saved successfully as {output_file}")
    plt.close()


def plot_record_section(weight_file, data_dir, freqmin, freqmax, t1, t2, output_file, component='Z', amp_scale=10.0):
    """
    Plots a record section of Vertical (Z) seismograms sorted by distance.
    Reads the stations and distances directly from an MTUQ weights.dat file.

    Args:
        weight_file (str): Path to the weights.dat file used to filter stations.
        data_dir (str): Directory containing the processed MTUQ SAC files.
        freqmin (float): Minimum frequency for the bandpass filter in Hz.
        freqmax (float): Maximum frequency for the bandpass filter in Hz.
        t1 (float): Start of the trim window (seconds relative to trace onset).
        t2 (float): End of the trim window (seconds relative to trace onset).
        output_file (str): Path and filename for the output PDF.
        amp_scale (float): Scaling factor for trace amplitudes to make them 
                           visible against the distance axis. Adjust as needed.

    Returns:
        None: Saves the record section plot as a PDF.
    """
    print(f"Preparing record section from {weight_file}...")

    # 1. Parse the weights.dat file to get the valid stations and their distances
    stations_to_plot = []
    if not os.path.exists(weight_file):
        print(f"Error: Weight file {weight_file} not found.")
        return

    with open(weight_file, 'r') as f:
        for line in f:
            parts = line.split()
            # MTUQ weight format: Prefix Distance Weight_Z ...
            if len(parts) >= 2:
                prefix = parts[0]
                try:
                    dist = float(parts[1])
                    stations_to_plot.append((dist, prefix))
                except ValueError:
                    continue

    # Sort the list so traces plot chronologically from bottom to top by distance
    stations_to_plot.sort()

    if not stations_to_plot:
        print("Error: No valid stations found in the weight file.")
        return

    # 2. Setup the Plot
    fig, ax = plt.subplots(figsize=(10, 12))

    # 3. Process and plot each station
    for dist, prefix in stations_to_plot:
        # MTUQ outputs usually use lowercase .z, but we check .Z just in case
        file_path = os.path.join(data_dir, f"{prefix}.{component.lower()}")
        if not os.path.exists(file_path):
            file_path = os.path.join(data_dir, f"{prefix}.{component.upper()}")
            if not os.path.exists(file_path):
                print(f"  -> Warning: Could not find {component} component for {prefix}. Skipping.")
                continue

        try:
            st = read(file_path)
            tr = st[0]
            
            # Remove mean, remove linear drift, and apply a 5% Hann taper
            tr.detrend("demean")
            tr.detrend("linear")
            tr.taper(max_percentage=0.05, type="hann")
            
            # Filter
            tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)

            # Trim relative to trace onset
            trace_onset = tr.stats.starttime
            tr.trim(trace_onset + t1, trace_onset + t2)

            if len(tr.data) == 0:
                print(f"  -> Warning: Trim window outside data bounds for {prefix}.")
                continue

            # Normalize the trace, then multiply by the scale factor
            max_amp = np.max(np.abs(tr.data))
            if max_amp == 0:
                continue
                
            norm_data = (tr.data / max_amp) * amp_scale
            times = tr.times() + t1

            # Plot: Y = Distance + Amplitude, X = Time
            ax.plot(times, dist + norm_data, color='black', linewidth=0.8)

            # Add the station name to the right edge of the trace for identification
            # Prefix format is usually EventID.Net.STA.Loc.Inst
            sta_name = prefix.split('.')[2] 
            ax.text(t2 + (0.02 * (t2 - t1)), dist, sta_name, verticalalignment='center', fontsize=8)

        except Exception as e:
            print(f"  X Error plotting {prefix}: {e}")

    # 4. Finalize and save
    ax.set_xlabel("Time (seconds) relative to trace onset")
    ax.set_ylabel("Distance (km)")
    ax.set_title(f"Record Section ({component.upper()} Component) | {freqmin}-{freqmax} Hz")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"  ✓ Record section saved successfully as {output_file}")
    plt.close()


def check_amplitude_decay(weight_file, data_dir, freqmin, freqmax, t1, t2, output_file, component='Z', scale='linear'):
    """
    Plots the maximum absolute amplitude of each station against its distance.
    This serves as a Quality Control (QC) check for instrument response removal.
    Outliers that do not follow the expected amplitude decay curve may have
    incorrect instrument responses or metadata.

    Args:
        weight_file (str): Path to the weights.dat file used to filter stations.
        data_dir (str): Directory containing the processed MTUQ SAC files.
        freqmin (float): Minimum frequency for the bandpass filter in Hz.
        freqmax (float): Maximum frequency for the bandpass filter in Hz.
        t1 (float): Start of the trim window (seconds relative to trace onset).
        t2 (float): End of the trim window (seconds relative to trace onset).
        output_file (str): Path and filename for the output PDF.
        component (str): Which component to evaluate (default: 'Z').
        scale (str): Y-axis scale, either 'linear' or 'log' (default: 'linear').

    Returns:
        None: Saves the QC scatter plot as a PDF.
    """
    print(f"Preparing amplitude decay QC plot from {weight_file}...")

    if not os.path.exists(weight_file):
        print(f"Error: Weight file {weight_file} not found.")
        return

    distances = []
    max_amplitudes = []
    station_labels = []

    # 1. Parse the weights.dat file
    with open(weight_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                prefix = parts[0]
                try:
                    dist = float(parts[1])
                except ValueError:
                    continue
                
                # 2. Find the file
                file_path = os.path.join(data_dir, f"{prefix}.{component.lower()}")
                if not os.path.exists(file_path):
                    file_path = os.path.join(data_dir, f"{prefix}.{component.upper()}")
                    if not os.path.exists(file_path):
                        print(f"  -> Warning: Could not find {component} component for {prefix}. Skipping.")
                        continue

                # 3. Read and process
                try:
                    st = read(file_path)
                    tr = st[0]
                    
                    # Preprocessing
                    tr.detrend("demean")
                    tr.detrend("linear")
                    tr.taper(max_percentage=0.05, type="hann")
                    
                    # Filter
                    tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)

                    # Trim
                    trace_onset = tr.stats.starttime
                    tr.trim(trace_onset + t1, trace_onset + t2)

                    if len(tr.data) == 0:
                        print(f"  -> Warning: Trim window outside data bounds for {prefix}.")
                        continue

                    # 4. Calculate Max Amplitude
                    max_amp = np.max(np.abs(tr.data))
                    if max_amp > 0:
                        distances.append(dist)
                        max_amplitudes.append(max_amp)
                        station_labels.append(prefix.split('.')[2]) # Extract STA name
                        
                except Exception as e:
                    print(f"  X Error processing {prefix}: {e}")

    if not distances:
        print("Error: No valid data points could be processed for the plot.")
        return

    # 5. Setup the Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the scatter points
    ax.scatter(distances, max_amplitudes, color='blue', edgecolor='black', s=50, alpha=0.7)

    # Add station labels to each point to easily identify outliers
    for i, txt in enumerate(station_labels):
        ax.annotate(txt, (distances[i], max_amplitudes[i]), fontsize=8, alpha=0.8, 
                    xytext=(5, 0), textcoords='offset points')

    # Formatting
    ax.set_xlabel("Distance (km)", fontsize=12)
    ax.set_ylabel(f"Maximum Absolute Amplitude ({component.upper()} Component)", fontsize=12)
    ax.set_title(f"Amplitude Decay QC | {freqmin}-{freqmax} Hz", fontsize=14)
    
    # Use the user-defined scale ('linear' or 'log')
    ax.set_yscale(scale)
    ax.grid(True, which="both", ls=":", alpha=0.6)

    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"  ✓ Amplitude QC plot saved successfully as {output_file}")
    plt.close()


def plot_station_map(weight_file, data_dir, output_file, component='Z', projection="M15c"):
    """
    Plots a map of the stations and the event epicenter using PyGMT.
    Coordinates are extracted dynamically from the SAC headers.

    Args:
        weight_file (str): Path to the weights.dat file used to filter stations.
        data_dir (str): Directory containing the processed MTUQ SAC files.
        output_file (str): Path and filename for the output PDF/PNG map.
        component (str): Which component to use to read headers (default: 'Z').
        projection (str): PyGMT projection string. Default is "M15c" (Mercator, 15cm).

    Returns:
        None: Saves the map to the output_file.
    """
    print(f"Preparing map from {weight_file}...")

    if not os.path.exists(weight_file):
        print(f"Error: Weight file {weight_file} not found.")
        return

    stlas = []
    stlos = []
    station_names = []
    
    evla = None
    evlo = None

    # 1. Parse the weights file and extract coordinates
    with open(weight_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                prefix = parts[0]
                
                # Find the file
                file_path = os.path.join(data_dir, f"{prefix}.{component.lower()}")
                if not os.path.exists(file_path):
                    file_path = os.path.join(data_dir, f"{prefix}.{component.upper()}")
                    if not os.path.exists(file_path):
                        continue

                try:
                    # headonly=True makes this incredibly fast
                    st = read(file_path, headonly=True)
                    header = st[0].stats.sac
                    
                    stlas.append(header.stla)
                    stlos.append(header.stlo)
                    station_names.append(prefix.split('.')[2])
                    
                    # We only need to grab the event coordinates once
                    if evla is None or evlo is None:
                        evla = header.evla
                        evlo = header.evlo
                        
                except Exception as e:
                    print(f"  X Error reading headers for {prefix}: {e}")

    if not stlas or evla is None:
        print("Error: Could not extract necessary coordinates from the SAC files.")
        return

    # 2. Calculate dynamic map region with padding
    pad = 2.0
    region = [
        min(stlos + [evlo]) - pad,
        max(stlos + [evlo]) + pad,
        min(stlas + [evla]) - pad,
        max(stlas + [evla]) + pad
    ]

    # 3. Setup and generate the PyGMT Plot
    fig = pygmt.Figure()
    
    # Basemap and coastlines
    fig.basemap(region=region, projection=projection, frame=True)
    fig.coast(shorelines="1/0.5p", land="lightgray", water="lightblue", borders="1/0.5p,gray")

    # Plot stations (Red triangles)
    fig.plot(x=stlos, y=stlas, style="t0.4c", fill="red", pen="1p,black")

    # Plot epicenter (Yellow star)
    fig.plot(x=evlo, y=evla, style="a0.8c", fill="yellow", pen="1p,black")

    # Add station names
    fig.text(x=stlos, y=stlas, text=station_names, justify="BL", offset="0.15c/0.15c", font="8p,Helvetica-Bold,black")

    fig.savefig(output_file)
    print(f"  ✓ Map saved successfully as {output_file}")