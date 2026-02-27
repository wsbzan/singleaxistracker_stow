import pandas as pd
import numpy as np
import pvlib
import matplotlib.pyplot as plt
from site_info import site_info as si
from api_calls import api_call

def build_site(
    axis_tilt,
    axis_azimuth,
    max_angle,
    backtrack,
    module_parameters,
    temperature_model_parameters,
    modules_per_string,
    strings_per_inverter,
    inverter_parameters,
    location
    ):
    '''
    Builds and returns pvlib modelchain and mount
    '''
    mount = pvlib.pvsystem.SingleAxisTrackerMount(
        axis_tilt=axis_tilt,
        axis_azimuth=axis_azimuth,
        max_angle=max_angle,
        backtrack=backtrack
    )
    # Array
    array = pvlib.pvsystem.Array(
        mount=mount,
        module_parameters=module_parameters,
        temperature_model_parameters = temperature_model_parameters,
        modules_per_string = modules_per_string,
        strings = strings_per_inverter
    )
    # System
    system = pvlib.pvsystem.PVSystem(
        arrays=[array], inverter_parameters=inverter_parameters
    )
    # Model Chain
    modelchain = pvlib.modelchain.ModelChain(
        system,
        location,
        ac_model = 'sandia',
        aoi_model='physical'
    )
    return modelchain, mount

def build_weather_data(
    psm4,
    tracker_angles,
    solar_position,
    gcr,
    axis_height,
    pitch,
    temperature_model_parameters,
    module_unit_mass
    ):
    '''
    Collects and returns weather data to run modelchain
    '''
    dni_extra = pvlib.irradiance.get_extra_radiation(solar_position.index)
    averaged_irradiance = pvlib.bifacial.infinite_sheds.get_irradiance_poa(
        tracker_angles['surface_tilt'].values, tracker_angles['surface_azimuth'].values,
        solar_position['apparent_zenith'].values, solar_position['azimuth'].values,
        gcr, axis_height, pitch,
        psm4['GHI'].values, psm4['DHI'].values, psm4['DNI'].values, psm4['Surface Albedo'].values,
        model='haysdavies', dni_extra=dni_extra)
    cell_temperature_steady_state = pvlib.temperature.faiman(
        poa_global=averaged_irradiance['poa_global'],
        temp_air=psm4['Temperature'],
        wind_speed=psm4['Wind Speed'],
        **temperature_model_parameters,
    )
    cell_temperature_steady_state.index = pd.to_datetime(cell_temperature_steady_state.index, format="mixed", errors='coerce')
    cell_temperature_steady_state.fillna(0,inplace=True)
    cell_temperature = pvlib.temperature.prilliman(
        cell_temperature_steady_state,
        psm4['Wind Speed'],
        unit_mass=module_unit_mass
    )
    x = pd.DataFrame({
        'poa_global': averaged_irradiance['poa_global'],
        'poa_direct': averaged_irradiance['poa_direct'],
        'poa_diffuse': averaged_irradiance['poa_diffuse'],
        'cell_temperature': cell_temperature,
        'precipitable_water': psm4['Precipitable Water'].values  # for the spectral model
    }) 
    return x

def recalculate_aoi_and_poa(
    tracker_angles_df,
    axis_tilt,
    axis_azimuth
    ):
    '''
    Takes the new tracker angles from stow conditions
    and recalculates aoi and poa at the new angle
    '''
    # need to read and reference the paper behind this function
    sol_pos = solar_position.copy()
    sol_pos.index = sol_pos.index.tz_localize(None)
    surface = pvlib.tracking.calc_surface_orientation(
            tracker_angles_df['stow_angle'], axis_tilt, axis_azimuth)
    aoi = pvlib.irradiance.aoi(surface['surface_tilt'],
        surface['surface_azimuth'], sol_pos['apparent_zenith'],
        sol_pos['azimuth'])
    aoi = aoi[~aoi.index.duplicated(keep='first')]
    tracker_angles_df['aoi'] = aoi
    tracker_angles_df['surface_tilt'] = surface['surface_tilt']
    tracker_angles_df['surface_azimuth'] = surface['surface_azimuth']
    return tracker_angles_df

def run_stow_conditions(
    df
    ):
    '''
    Adjusts tracker angles based on stow conditions
    '''
    df.insert(0,'trigger',np.nan)
    df['trigger'] = df['trigger'].astype('object')  # Ensure 'trigger' column is of object type to hold string values
    df.insert(1,'stow_angle',np.nan)
    df.insert(2,'stow_setpoint',np.nan)
    df['td'] = pd.to_datetime(df.index)
    df['time_delta'] = (df['td']-df['td'].shift())
    relaxation_factor = 0
    # Set Initial Tracker Angle and Setpoint
    df.at[df.index[0], 'stow_setpoint'] = df.at[df.index[0], 'tracker_theta']
    df.at[df.index[0], 'stow_angle'] = df.at[df.index[0], 'tracker_theta']
    max_angle = 60
    for i in range(len(df)-1):
        idx = df.index[i]
        next_idx = df.index[i+1]
        row = df.loc[idx]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        # Stow Conditions
        # Storm
        if row['Wind Speed (m/s)'] > 25 or\
            row['Weather Code'] in [200,201,202,230,231,232,233,511]:
        # ['Thunderstorm with light rain','Thunderstorm with rain','Thunderstorm with heavy rain','Thunderstorm with light drizzle',\
        # 'Thunderstorm with drizzle','Thunderstorm with heavy drizzle','Thunderstorm with Hail']:
            if pd.isna(df.at[idx, 'trigger']):
                df.at[idx, 'trigger'] = "Storm"
                relaxation_factor = 40
                if row['stow_angle'] < 0:
                    df.at[next_idx, 'stow_setpoint'] = - max_angle
                else:
                    df.at[next_idx, 'stow_setpoint'] = max_angle
            else:
                pass
                # already triggered, but I think I want to track that
        # Wind
        elif row['Wind Speed (m/s)'] > 11 or row['Wind Gust Speed (m/s)'] > 16:
            if pd.isna(df.at[idx, 'trigger']):
                df.at[idx, 'trigger'] = "Wind"
                relaxation_factor = 20
                if row['stow_angle'] < 0:
                    df.at[next_idx, 'stow_setpoint'] = -40
                else:
                    df.at[next_idx, 'stow_setpoint'] = 40
            else:
                pass
                # already triggered, but I think I want to track that
        # Snow
        elif row['Snowfall Rate (mm/hr)'] > 0 or \
        row['Weather Code'] in [600,601,602,610,611,612,621,622,623]:
        # ['Freezing Rain','Snow','Heavy Snow','Mix snow/rain','Sleet','Snow Shower','Heavy snow shower','Flurries']:
            if pd.isna(df.at[idx, 'trigger']):
                df.at[idx, 'trigger'] = "Snow"
                relaxation_factor = 30
                # Determine Wind Direction
                # If Wind from East, Stow to East
                if row['Wind Dir (Deg)'] <=180:
                    df.at[next_idx, 'stow_setpoint'] = - max_angle
                # If Wind from West, Stow to West  
                else:
                    df.at[next_idx, 'stow_setpoint'] = max_angle
            else:
                pass
                # already triggered, but I think I want to track that
        # If no active trigger, check if relaxation factor is > 0
        elif relaxation_factor > 0:
            df.at[idx, 'trigger'] = "Relaxing"
            time_delta = df.at[idx, 'time_delta'].total_seconds() / 60
            relaxation_factor -= time_delta
            if relaxation_factor > 0:
                df.at[next_idx, 'stow_setpoint'] = df.at[idx, 'stow_setpoint']
            else:
                df.at[next_idx, 'stow_setpoint'] = df.at[next_idx, 'tracker_theta']
        else:
            df.at[next_idx, 'stow_setpoint'] = df.at[next_idx, 'tracker_theta']
            df.at[idx, 'trigger'] = "Ideal Tracking"
        # Determine delta between actual angle(idx) and setpoint angle (idx+1)
        stow_setpoint_val = df['stow_setpoint'].loc[next_idx]
        if isinstance(stow_setpoint_val, pd.Series):
            setpoint_angle = stow_setpoint_val.iloc[0]
        else:
            setpoint_angle = stow_setpoint_val
        
        stow_angle_val = df['stow_angle'].loc[idx]
        if isinstance(stow_angle_val, pd.Series):
            current_angle = stow_angle_val.iloc[0]
        else:
            current_angle = stow_angle_val
        
        angle_delta = setpoint_angle - current_angle
        # 20 degrees per time step (15 minutes)
        max_angle_change = 30
        if abs(angle_delta) > max_angle_change:
            # Update stow angle for next time step
            if (setpoint_angle < current_angle):
                df.at[next_idx, 'stow_angle'] = (df.at[idx, 'stow_angle'] - max_angle_change).clip(-max_angle, max_angle)
            else:
                df.at[next_idx, 'stow_angle'] = (df.at[idx, 'stow_angle'] + max_angle_change).clip(-max_angle, max_angle)
        else:
            df.at[next_idx, 'stow_angle'] = df.at[next_idx, 'stow_setpoint']

    return df

if __name__ == '__main__':
    # Phase A
    # Imports and Instantiations
    # Site Info imported from site_info.py
    # Establish Date Range -> pandas DateTimeIndex
    input_df = pd.read_csv('psmv4_weatherbit.csv', parse_dates=True)
    input_df['Timestamp (UTC)'] = pd.to_datetime(input_df['Timestamp (UTC)'])
    input_df.set_index('Timestamp (UTC)', inplace=True)
    times = pd.date_range(si['start'], si['end'], freq=si['freq']) #, tz=si['tz'])
    # Establish PVlib Location Object
    location = pvlib.location.Location(
        latitude=si['latitude'],
        longitude=si['longitude'],
        tz=si['tz'],
        altitude=si['altitude'],
        name=si['name']
    )
    # Solar Position based off location and times
    solar_position = location.get_solarposition(input_df.index)
    # Import Weather Data from PSM4
    # if si['psm']:
    #     api = api_call()
    #     psm4_data = api.fetch_psm4_data()
    #     with open('psm4_data.csv', 'w') as file:
    #         file.write(psm4_data)
    #     w_df = pd.read_csv('psm4_data.csv', index_col=0, parse_dates=True)
    # else:
    #     w_df = pd.read_csv(si['psm_file'], parse_dates=True,skiprows=2)
    #     w_df['timestamp_local'] = pd.to_datetime(w_df[['Year','Month','Day','Hour','Minute']])
    #     w_df.set_index('timestamp_local', inplace=True)
    # # Import Stow Weather Data from Weatherbit
    # if si['weaterbit']:
    #     api = api_call()
    #     weatherbit_data = api.fetch_weatherbit_data()
    #     df = pd.json_normalize(weatherbit_data, 'data',["city_id","city_name","country_code","lat","lon","state_code","station_id","timezone"])
    #     df.set_index(pd.to_datetime(df['timestamp_local']), inplace=True)
    #     df.rename(columns={
    #         'temp':'temp_air',
    #         'Wind Speed (m/s)':'wind_speed',
    #         'precipitable_water':'precipitable_water'
    #     }, inplace=True)
    #     sw_df = df[['temp_air','wind_speed','precipitable_water']]
    #     sw_df.to_csv('weatherbit_data.csv')
    # else:
    #     sw_df = pd.read_csv(si['weatherbit_file'], parse_dates=True)
    #     # sw_df['Timestamp (Local)'] = pd.to_datetime(sw_df['Timestamp (Local)'])
    #     # sw_df.loc[6632:29476, 'Timestamp (Local)'] -= pd.Timedelta(hours=1)
    #     sw_df.set_index(pd.to_datetime(sw_df['Timestamp (Local)']), inplace=True)
    # Phase B
    # Establish PVLIB Parameters, Modelchain, and Mount
    # Temperature model
    temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
    # PV module and inverter models (use realistic specs)
    cec_module_db = pvlib.pvsystem.retrieve_sam('cecmod')
    module_parameters = cec_module_db[si['module_name']]
    # ensure that correct spectral correction is applied
    module_parameters['Technology'] = 'CdTe'
    cec_inverter_db = pvlib.pvsystem.retrieve_sam('cecinverter')
    inverter_parameters = cec_inverter_db[si['inverter__name']]
    # Build Site Modelchain and Mount
    mc, mount = build_site(si['axis_tilt'], si['axis_azimuth'], si['max_angle'], si['backtrack'],
        module_parameters,si['temperature_model_parameters'], si['modules_per_string'],
        si['strings_per_inverter'], inverter_parameters, location)
    # Phase C
    # Stow Conditions
    # Get Ideal Tracker Angles
    tracker_angles_1 = mount.get_orientation(
        solar_position['apparent_zenith'],
        solar_position['azimuth'])
    tracker_angles_1['tracker_theta'] = tracker_angles_1['tracker_theta'].ffill()
    # Copy Ideal Angles over to Stow Angles
    # tracker_angles_2 = tracker_angles_1.copy()
    # Get Stow Conditions Angles
    tracker_df = pd.concat([tracker_angles_1, input_df],axis=1)
    # tracker_df = tracker_df[~tracker_df.index.duplicated(keep='first')]
    tracker_angles_2 = run_stow_conditions(tracker_df)
    tracker_angles_2.to_csv('tracker_angles_with_stow_conditions.csv')
    vplotnum=1
    if vplotnum == 1:
        vplot = tracker_angles_2.loc[(tracker_angles_2['timestamp_local'].str.startswith('4/2/24')) |
                            (tracker_angles_2['timestamp_local'].str.startswith('4/3/24')) |
                            (tracker_angles_2['timestamp_local'].str.startswith('4/3/25'))]
    tracker_df.to_csv('tracker_df_with_stow_conditions_and_weather.csv')
    vplot.to_csv('april 2 and 3 example.csv')
    # Recalculate AOI for Stow Angles
    # need to read and reference the paper behind this function
    # need to rename to remove poa as not part of below function
    tracker_angles_2 = recalculate_aoi_and_poa(tracker_angles_2,
                        si['axis_tilt'], si['axis_azimuth'])
    # Phase D
    # Build Weather Data and Estimate Power Output
    # Build weather data using different tracker angles to get POA
    results = []
    ii=0
    for i in [tracker_angles_1, tracker_angles_2]:
        ii +=1
        wd = build_weather_data(input_df, i, solar_position, si['gcr'],
            si['axis_height'], si['pitch'], si['temperature_model_parameters'], si['module_unit_mass'])
        wd.to_csv('wd'+str(ii)+'.csv')
        mc.run_model_from_poa(wd)
        ac = mc.results.ac / 1000
        dc = mc.results.dc['p_mp'] / 1000
        results.append([ac, dc])

    # Map each unique trigger to a color
    unique_triggers = tracker_df['trigger'].unique()
    color_map = dict(zip(unique_triggers, plt.cm.tab20.colors[:len(unique_triggers)]))

    # Prepare the plot
    fig, ax = plt.subplots(figsize=(12, 2))

    # Plot each timestamp as a colored bar
    tracker_angles_2['trigger'].fillna("Ideal Tracking", inplace=True)  # Fill NaN values with "Ideal Tracking"
    # Remove any remaining NaNs in 'trigger' column
    tracker_angles_2 = tracker_angles_2[~tracker_angles_2['trigger'].isna()]

    # Manually specify color mapping for triggers
    manual_color_map = {
        'Ideal Tracking': 'gray',
        'Storm': 'red',
        'Wind': 'blue',
        'Snow': 'cyan',
        # Add or adjust colors as needed
    }

    for i, (timestamp, row) in enumerate(tracker_angles_2.iterrows()):
        trigger = row['trigger']
        color = manual_color_map.get(trigger, 'black')  # Default to black if not found
        ax.bar(timestamp, 1, color=color, width=0.01)  # Adjust width as needed

    # Set x-axis as time
    ax.set_xlim(tracker_angles_2.index.min(), tracker_angles_2.index.max())
    ax.set_xlabel('Time')
    ax.set_yticks([])  # Hide y-axis

    # Create legend
    handles = [plt.Line2D([0], [0], color=manual_color_map.get(trig, 'black'), lw=4) for trig in manual_color_map.keys()]
    ax.legend(handles, list(manual_color_map.keys()), title='Trigger', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()
    plt.savefig('stow_conditions_timeline.png', dpi=300, bbox_inches='tight')

    print('Total energy output DC (kWh) - True Tracking:', results[0][1].sum())
    print('Total energy output DC (kWh) - Sample Wind Stow:', results[1][1].sum())

    results[0][1].to_csv('true_tracking_dc_output.csv')
    results[1][1].to_csv('stow_tracking_dc_output.csv')
    tracker_angles_2.to_csv('tracker_angles_with_triggers.csv')

    print(results)