"""
Seismic data acquisition module.

Handles querying FDSN clients, managing channel fallbacks (preferring HH over BH),
downloading MiniSEED data, healing gaps, and plotting station maps.
"""

import os
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pygmt

def search_data(client, network, latitude, longitude, maxradius, starttime, endtime):
    """
    Searches for stations within a radius and filters for complete 3-component sets.
    
    Prefers 'HH' broadband channels over 'BH'. If a station does not have a 
    complete 3-component set (Z, N/1, E/2) for either, it is skipped.

    Args:
        client (obspy.clients.fdsn.Client): The initialized FDSN client.
        network (str): Network code (e.g., "*", "IU").
        latitude (float): Event latitude.
        longitude (float): Event longitude.
        maxradius (float): Maximum search radius in degrees.
        starttime (obspy.UTCDateTime): Start of the search window.
        endtime (obspy.UTCDateTime): End of the search window.

    Returns:
        tuple: (networks_found, stations_found, latitudes, longitudes, selected_channels)
    """
    networks_found = []
    stations_found = []
    latitudes = []
    longitudes = []
    selected_channels = [] 

    try:
        inventory = client.get_stations(network=network, latitude=latitude, longitude=longitude,
                                        maxradius=maxradius, starttime=starttime, endtime=endtime, level="channel")
        print("Stations retrieved successfully!")

        for net in inventory:
            for station in net:
                hh_channels = {chan.code for chan in station if chan.code in ["HHZ", "HHE", "HHN", "HH1", "HH2"]}
                bh_channels = {chan.code for chan in station if chan.code in ["BHZ", "BHE", "BHN", "BH1", "BH2"]}
                
                valid_sets = []
                
                if "BHZ" in bh_channels and ({"BHE", "BHN"}.issubset(bh_channels) or {"BH1", "BH2"}.issubset(bh_channels)):
                    valid_sets.append(list(bh_channels))
                    
                if "HHZ" in hh_channels and ({"HHE", "HHN"}.issubset(hh_channels) or {"HH1", "HH2"}.issubset(hh_channels)):
                    valid_sets.append(list(hh_channels))

                if not valid_sets:
                    continue
                
                networks_found.append(net.code)
                stations_found.append(station.code)
                latitudes.append(station.latitude)
                longitudes.append(station.longitude)
                selected_channels.append(valid_sets)

                print(f"Added: {net.code}.{station.code} - Available Sets: {valid_sets}")

    except Exception as e:
        print(f"An error occurred during search: {e}")
    
    return networks_found, stations_found, latitudes, longitudes, selected_channels

def make_map(stations, latitudes, longitudes, event_lat, event_lon, filename):
    """Generates a PyGMT map of the stations and event epicenter."""
    fig = pygmt.Figure()
    region = [min(longitudes) - 2, max(longitudes) + 2, min(latitudes) - 2, max(latitudes) + 2]

    fig.basemap(region=region, projection="M6i", frame=True)
    fig.coast(shorelines=True, water="lightblue", land="lightgray")
    fig.plot(x=longitudes, y=latitudes, style="t0.3c", color="red", pen="black")
    fig.plot(x=event_lon, y=event_lat, style="a0.5c", color="yellow", pen="black")

    for lon, lat, station in zip(longitudes, latitudes, stations):
        fig.text(x=lon, y=lat, text=station, font="8p,Helvetica-Bold,black", justify="BL", offset="0.1c/0.1c")

    fig.savefig(filename)
    print(f"Map saved as {filename}")

def download_event_data(client, networks, stations, channel_sets_list, starttime, endtime, out_dir="event_data"):
    """Downloads waveforms, heals gaps via interpolation, and saves MiniSEED/XML."""
    os.makedirs(out_dir, exist_ok=True)
    log_file_path = os.path.join(out_dir, "interpolation_log.txt")

    for net, sta, sta_channel_sets in zip(networks, stations, channel_sets_list):
        download_success = False
        
        for channel_set in sta_channel_sets:
            if download_success:
                break 

            channel_str = ",".join(channel_set)
            print(f"\nAttempting {net}.{sta} with channels: {channel_str}...")

            try:
                st = client.get_waveforms(network=net, station=sta, location="*",
                                          channel=channel_str, starttime=starttime, endtime=endtime)
                
                found_channels = set([tr.stats.channel for tr in st])
                if len(found_channels) < 3:
                    print(f"  -> Warning: Only received {len(found_channels)}/3 channels. Trying fallback...")
                    continue 
                
                gaps = st.get_gaps()
                if gaps:
                    with open(log_file_path, "a") as logfile:
                        logfile.write(f"\n--- Gaps patched for {net}.{sta} ({channel_str}) ---\n")
                        for gap in gaps:
                            logfile.write(f"Interpolated {gap[6]:.3f} seconds between {gap[4]} and {gap[5]}\n")

                st.merge(method=1, fill_value='interpolate')
                
                for tr in st:
                    loc_str = tr.stats.location if tr.stats.location else "--"
                    filename = f"{net}_{sta}_{loc_str}_{tr.stats.channel}_{starttime.strftime('%Y%m%dT%H%M%S')}.miniseed"
                    file_path = os.path.join(out_dir, filename)
                    st.select(id=tr.id).write(file_path, format="MSEED")
                    print(f"  ✓ Saved: {filename}")

                station_xml_path = os.path.join(out_dir, f"{net}_{sta}.xml")
                if not os.path.exists(station_xml_path):
                    inventory = client.get_stations(network=net, station=sta, 
                                                    starttime=starttime, endtime=endtime, level="response")
                    inventory.write(station_xml_path, format="STATIONXML")

                download_success = True 

            except Exception as e:
                print(f"  -> Failed to get {channel_str}: {e}")
                
        if not download_success:
            print(f"X Could not find complete data for {net}.{sta} after trying all options.")


def run_download(config_dict):
    """
    Main execution wrapper for the download process.
    
    Args:
        config_dict (dict): The loaded JSON configuration dictionary.
    """
    client = Client(config_dict["client_name"])

    event_time = UTCDateTime(config_dict["event"]["time"])
    event_lat = config_dict["event"]["latitude"]
    event_lon = config_dict["event"]["longitude"]

    network_query = config_dict["search"]["network_query"]
    max_distance_degrees = config_dict["search"]["max_distance_degrees"]
    minutes_before = config_dict["search"]["minutes_before"]
    minutes_after = config_dict["search"]["minutes_after"]

    starttime = event_time - (minutes_before * 60)
    endtime = event_time + (minutes_after * 60)

    print(f"Searching for stations within {max_distance_degrees} degrees of {event_lat}, {event_lon}...")
    print(f"Time Window: {starttime} to {endtime}")

    networks, stations, latitudes, longitudes, selected_channels = search_data(
        client, network_query, event_lat, event_lon, max_distance_degrees, starttime, endtime
    )

    if stations:
        print(f"\nFound {len(stations)} valid stations. Starting download...")
        out_dir = f"{event_time.strftime('%Y%m%dT%H%M%S')}_data"
        download_event_data(client, networks, stations, selected_channels, starttime, endtime, out_dir=out_dir)
        
        # filename = f"{event_time.strftime('%Y%m%dT%H%M%S')}_Station_Map.pdf"
        # make_map(stations, latitudes, longitudes, event_lat, event_lon, filename)
    else:
        print("No stations found matching the criteria.")