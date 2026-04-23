"""
Seismic data processing module for MTUQ.

Handles reading MiniSEED, removing instrument response, downsampling,
rotating horizontals to Radial/Transverse, and injecting MTUQ-required SAC headers.
"""

import os
import glob
from obspy import read, read_inventory, UTCDateTime
from obspy.geodetics import gps2dist_azimuth

def get_event_id(origin_time):
    """Generates the 17-digit MTUQ Event ID (YYYYMMDDHHMMSS000)."""
    return origin_time.strftime('%Y%m%d%H%M%S000')

def get_mtuq_filename(event_id, net, sta, loc, chan):
    """Formats the filename: EventID.Network.Station.Location.Instrument.component."""
    inst = chan[:2]          
    comp = chan[-1].lower()  
    return f"{event_id}.{net}.{sta}.{loc}.{inst}.{comp}"

def inject_mtuq_sac_headers(tr, event_lat, event_lon, event_depth_km, sta_lat, sta_lon, sta_elev, dist_km, az, baz, origin_time):
    """Populates SAC headers and shifts the reference time to the Earthquake Origin Time."""
    if not hasattr(tr.stats, 'sac'):
        tr.stats.sac = {}

    tr.stats.sac['evla'] = event_lat
    tr.stats.sac['evlo'] = event_lon
    tr.stats.sac['evdp'] = event_depth_km
    tr.stats.sac['stla'] = sta_lat
    tr.stats.sac['stlo'] = sta_lon
    tr.stats.sac['stel'] = sta_elev
    tr.stats.sac['dist'] = dist_km
    tr.stats.sac['az'] = az
    tr.stats.sac['baz'] = baz
    tr.stats.sac['lcalda'] = 1 
    tr.stats.sac['iztype'] = 11  
    tr.stats.sac['nzyear'] = origin_time.year
    tr.stats.sac['nzjday'] = origin_time.julday
    tr.stats.sac['nzhour'] = origin_time.hour
    tr.stats.sac['nzmin'] = origin_time.minute
    tr.stats.sac['nzsec'] = origin_time.second
    tr.stats.sac['nzmsec'] = int(origin_time.microsecond / 1000)

    time_offset = tr.stats.starttime - origin_time
    tr.stats.sac['b'] = time_offset
    tr.stats.sac['e'] = time_offset + (tr.stats.npts - 1) * tr.stats.delta
    tr.stats.sac['o'] = 0.0  
    
    return tr


def run_processing(config_dict):
    """
    Main execution wrapper for the MTUQ processing pipeline.
    
    Args:
        config_dict (dict): The loaded JSON configuration dictionary.
    """
    origin_time_str = config_dict["event"]["time"]
    origin_time = UTCDateTime(origin_time_str)
    
    data_dir = f"{origin_time.strftime('%Y%m%dT%H%M%S')}_data"
    out_dir = f"{data_dir}_SAC_MTUQ"
    os.makedirs(out_dir, exist_ok=True)
    
    event_lat = config_dict["event"]["latitude"]
    event_lon = config_dict["event"]["longitude"]
    event_depth_km = config_dict["event"]["depth_km"] 
    pre_filt = config_dict["processing"]["pre_filter"]
    sample_rate = config_dict["processing"]["sample_rate"]
    
    log_file_path = os.path.join(out_dir, "processing.log")
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
        
    def lprint(message):
        print(message)
        with open(log_file_path, "a") as logf:
            logf.write(message + "\n")
    
    event_id = get_event_id(origin_time)
    weight_file_path = os.path.join(out_dir, "weights.dat")
    weights_dict = {}

    padded_stations = set()
    downsampled_stations = {}
    low_sr_stations = {}
    full_3comp_stations = []
    z_only_stations = []
    h_only_stations = []
    skipped_stations = []

    xml_files = glob.glob(os.path.join(data_dir, "*.xml"))
    lprint(f"Starting MTUQ Pipeline for Event ID: {event_id}")

    for xml_file in xml_files:
        basename = os.path.basename(xml_file).replace(".xml", "")
        try:
            net, sta = basename.split("_")
        except ValueError:
            continue

        mseed_pattern = os.path.join(data_dir, f"{net}_{sta}_*.miniseed")
        mseed_files = glob.glob(mseed_pattern)
        if not mseed_files:
            continue

        lprint(f"\nProcessing {net}.{sta}...")
        try:
            st = read(mseed_pattern)
            inv = read_inventory(xml_file)
            st.merge(method=1, fill_value='interpolate')

            locations = list(set([tr.stats.location for tr in st]))

            for loc in locations:
                loc_str = loc if loc else "" 
                sta_loc_id = f"{sta}.{loc_str}" if loc_str else f"{sta}.--"
                lprint(f"  -> Location code: '{loc_str}'")

                st_loc = st.select(location=loc).copy()
                if len(st_loc) == 0:
                    continue
                
                channels = [tr.stats.channel for tr in st_loc]
                inst_code = channels[0][:2] if channels else "XX"
                
                has_z = any(c.endswith(('Z', 'z')) for c in channels)
                has_n = any(c.endswith(('N', 'n', '1')) for c in channels)
                has_e = any(c.endswith(('E', 'e', '2')) for c in channels)
                has_horizontals = has_n and has_e

                if not has_z and not has_horizontals:
                    lprint(f"     X Skipped: Missing Vertical and complete Horizontal components.")
                    skipped_stations.append(sta_loc_id)
                    continue

                if has_z and has_horizontals:
                    weight_str = "1 1   1 1 1"
                    status = "FULL"
                elif has_z and not has_horizontals:
                    weight_str = "1 0   1 0 0"
                    status = "Z_ONLY"
                    lprint(f"     ! Notice: Missing horizontal pair. Proceeding with Vertical (Z) only.")
                elif not has_z and has_horizontals:
                    weight_str = "0 1   0 1 1"
                    status = "H_ONLY"
                    lprint(f"     ! Notice: Missing Vertical (Z). Proceeding with Horizontals (R/T) only.")

                npts_list = [tr.stats.npts for tr in st_loc]
                if len(set(npts_list)) > 1:
                    padded_stations.add(sta_loc_id)
                    min_start = min([tr.stats.starttime for tr in st_loc])
                    max_end = max([tr.stats.endtime for tr in st_loc])
                    st_loc.trim(starttime=min_start, endtime=max_end, pad=True, fill_value=0)
                    lprint(f"     + Padded components with zeros to perfectly match lengths.")

                for tr in st_loc:
                    current_sr = round(tr.stats.sampling_rate, 2)
                    if current_sr > sample_rate:
                        tr.resample(sample_rate)
                        downsampled_stations[sta_loc_id] = (current_sr, sample_rate)
                        lprint(f"     ↓ Downsampled {tr.stats.channel} from {current_sr} Hz to {sample_rate} Hz")
                    elif current_sr < sample_rate:
                        low_sr_stations[sta_loc_id] = current_sr
                        lprint(f"     ! WARNING: {tr.stats.channel} sampling rate ({current_sr} Hz) is below target ({sample_rate} Hz).")

                sta_lat = inv[0][0].latitude
                sta_lon = inv[0][0].longitude
                sta_elev = inv[0][0].elevation
                dist_m, az, baz = gps2dist_azimuth(event_lat, event_lon, sta_lat, sta_lon)
                dist_km = dist_m / 1000.0

                st_loc.remove_response(inventory=inv, pre_filt=pre_filt, output="VEL", water_level=60)
                
                prefix = f"{event_id}.{net}.{sta}.{loc_str}.{inst_code}"
                if prefix not in weights_dict:
                    weights_dict[prefix] = (dist_km, weight_str)

                if has_z:
                    st_z = st_loc.select(component="Z")
                    for tr in st_z:
                        tr = inject_mtuq_sac_headers(tr, event_lat, event_lon, event_depth_km, 
                                                     sta_lat, sta_lon, sta_elev, dist_km, az, baz, origin_time)
                        out_name = get_mtuq_filename(event_id, net, sta, loc_str, tr.stats.channel)
                        tr.write(os.path.join(out_dir, out_name), format="SAC")
                    lprint(f"     ✓ Saved Z component")

                if has_horizontals:
                    st_h = st_loc.copy()
                    try:
                        st_h.rotate(method="->ZNE", inventory=inv)
                        st_h.rotate(method="NE->RT", back_azimuth=baz)

                        for tr in st_h:
                            if tr.stats.channel[-1] in ['R', 'T']:
                                tr = inject_mtuq_sac_headers(tr, event_lat, event_lon, event_depth_km, 
                                                             sta_lat, sta_lon, sta_elev, dist_km, az, baz, origin_time)
                                out_name = get_mtuq_filename(event_id, net, sta, loc_str, tr.stats.channel)
                                tr.write(os.path.join(out_dir, out_name), format="SAC")
                        lprint(f"     ✓ Saved R, T components")
                        
                        if status == "FULL":
                            full_3comp_stations.append(sta_loc_id)
                        else:
                            h_only_stations.append(sta_loc_id)

                    except Exception as e:
                        lprint(f"     X Rotation failed: {e}. Horizontals discarded for {sta_loc_id}.")
                        if has_z:
                            z_only_stations.append(sta_loc_id)
                        else:
                            skipped_stations.append(sta_loc_id)
                else:
                    if has_z:
                        z_only_stations.append(sta_loc_id)

        except Exception as e:
            lprint(f"  X Error processing {net}.{sta}: {e}")

    sorted_weights = sorted(weights_dict.items(), key=lambda item: item[1][0])
    with open(weight_file_path, "w") as wf:
        for prefix, (dist_km, weight_str) in sorted_weights:
            line = f"{prefix:<32} {dist_km:^8.2f}   {weight_str}     0.00 0     0.00 0   0\n"
            wf.write(line)

    lprint("\n==========================================")
    lprint("          PROCESSING SUMMARY              ")
    lprint("==========================================")
    lprint(f"\n[COMPONENT COMPLETENESS]")
    lprint(f"  Full 3-Components (Z, R, T): {len(full_3comp_stations)} stations")
    lprint(f"  Vertical (Z) Only: {len(z_only_stations)} stations")
    lprint(f"  Horizontals (R, T) Only: {len(h_only_stations)} stations")
    lprint(f"  Skipped (Missing Data/Failed): {len(skipped_stations)} stations")
    lprint(f"\nMTUQ Processing complete! Check the '{out_dir}' folder.")
    lprint(f"Weights generated at: {weight_file_path}")