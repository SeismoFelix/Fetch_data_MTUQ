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

def plot_station_data(station_id, data_dir, origin_time_str, freqmin, freqmax, t1, t2, phases, output_file):
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

    Returns:
        None: Saves the plot as a PDF to the specified output_file.
    """
    print(f"Preparing plot for {station_id}...")

    # Read the 3 components (Z, R, T) utilizing glob-style wildcards
    try:
        st = read(os.path.join(data_dir, f"{station_id}_*Z.sac"))
        st += read(os.path.join(data_dir, f"{station_id}_*R.sac"))
        st += read(os.path.join(data_dir, f"{station_id}_*T.sac"))
    except Exception as e:
        print(f"Error: Could not read files for {station_id}: {e}")
        return

    if len(st) != 3:
        print(f"Error: Found {len(st)} components. Expected exactly 3 (Z, R, T).")
        return

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

    # Calculate phase arrivals using the ak135 1D velocity model
    model = TauPyModel(model="ak135")
    arrivals = model.get_travel_times(source_depth_in_km=evdp_km, 
                                      distance_in_degree=dist_deg, 
                                      phase_list=phases)

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