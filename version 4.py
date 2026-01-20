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
    psm3,
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
    dni_extra = pvlib.irradiance.get_extra_radiation(psm3.index)
    averaged_irradiance = pvlib.bifacial.infinite_sheds.get_irradiance_poa(
        tracker_angles['surface_tilt'], tracker_angles['surface_azimuth'],
        solar_position['apparent_zenith'], solar_position['azimuth'],
        gcr, axis_height, pitch,
        psm3['ghi'], psm3['dhi'], psm3['dni'], psm3['albedo'],
        model='haydavies', dni_extra=dni_extra,
    )
    cell_temperature_steady_state = pvlib.temperature.faiman(
        poa_global=averaged_irradiance['poa_global'],
        temp_air=psm3['temp_air'],
        wind_speed=psm3['wind_speed'],
        **temperature_model_parameters,
    )
    cell_temperature = pvlib.temperature.prilliman(
        cell_temperature_steady_state,
        psm3['wind_speed'],
        unit_mass=module_unit_mass
    )
    return pd.DataFrame({
        'poa_global': averaged_irradiance['poa_global'],
        'poa_direct': averaged_irradiance['poa_direct'],
        'poa_diffuse': averaged_irradiance['poa_diffuse'],
        'cell_temperature': cell_temperature,
        'precipitable_water': psm3['precipitable_water']  # for the spectral model
    })

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
    surface = pvlib.tracking.calc_surface_orientation(
            tracker_angles_df['tracker_theta'], axis_tilt, axis_azimuth)
    aoi = pvlib.irradiance.aoi(surface['surface_tilt'],
        surface['surface_azimuth'], solar_position['apparent_zenith'],
        solar_position['azimuth'])
    tracker_angles_df['aoi'] = aoi
    tracker_angles_df['surface_tilt'] = surface_tilt
    tracker_angles_df['surface_azimuth'] = surface_azimuth
    return tracker_angles_df

def run_stow_conditions(
    df
    ):
    '''
    Adjusts tracker angles based on stow conditions
    '''
    df.add_column('trigger',np.nan,inplace=True)
    df.add_column('stow_angle',np.nan,inplace=True)
    df.add_column('stow_setpoint',np.nan,inplace=True)
    relaxation_factor = 0
    # Set Initial Tracker Angle and Setpoint
    df.at[df.index[0], 'stow_setpoint'] = df.loc[df.index[0], 'tracker_theta']
    df.at[df.index[0], 'stow_angle'] = df.loc[df.index[0], 'tracker_theta']
    for idx, row in df.iterrows():
        # Stow Conditions
        # Check trigger
        if df.loc[idx - 1, 'trigger'] is not np.nan and relaxation_factor > 0:
            time_delta = (idx - (idx - 1)).total_seconds() / 3600
            relaxation_factor -= time_delta
            if relaxation_factor == 0:
                continue

        # Storm

        # Hail

        # Wind
        # X1 and X2
        if row['wind_speed'] > 10 or row['wind_gust_spd'] > 20:
            df.loc[idx, 'trigger'] = "Wind"
            relaxation_factor = 20
            # T1
            if row['stow_angle'] < 0:
                df.at[idx+1, 'stow_setpoint'] = -40
            else:
                df.at[idx+1, 'stow_setpoint'] = 40
        # Snow

        # Flood

        # Determine delta between actual angle(idx) and setpoint angle (idx+1)
        angle_delta = df.loc[idx+1, 'stow_setpoint'] - df.loc[idx, 'stow_angle']
        # 20 degrees per time step (15 minutes)
        max_angle_change = 20
        if abs(angle_delta) > max_angle_change:
            angle_delta = np.sign(angle_delta)
            # Update stow angle for next time step
            df.at[idx+1, 'stow_angle'] = df.loc[idx, 'stow_angle'] + angle_delta
        else:
            df.at[idx+1, 'stow_angle'] = df.loc[idx+1, 'stow_setpoint']

    return df['tracker_theta']

if __name__ == '__main__':
    # Phase A
    # Imports and Instantiations
    # Site Info imported from site_info.py
    # Establish Date Range -> pandas DateTimeIndex
    times = pd.date_range(si['start'], si['end'], freq=si['freq'], tz=si['tz'])
    # Establish PVlib Location Object
    location = pvlib.location.Location(
        latitude=si['latitude'],
        longitude=si['longitude'],
        tz=si['tz'],
        altitude=si['altitude'],
        name=si['name']
    )
    # Solar Position based off location and times
    solar_position = location.get_solarposition(times)
    # Import Weather Data from PSM4
    if si['psm']:
        api = api_call()
        psm4_data = api.fetch_psm4_data()
        with open('psm4_data.csv', 'w') as file:
            file.write(psm4_data)
        w_df = pd.read_csv('psm4_data.csv', index_col=0, parse_dates=True)
    else:
        w_df = pd.read_csv(si['psm_file'], index_col=0, parse_dates=True)
    # Import Stow Weather Data from Weatherbit
    if si['weaterbit']:
        api = api_call()
        weatherbit_data = api.fetch_weatherbit_data()
        df = pd.json_normalize(weatherbit_data, 'data',["city_id","city_name","country_code","lat","lon","state_code","station_id","timezone"])
        df.set_index(pd.to_datetime(df['timestamp_utc']), inplace=True)
        df.rename(columns={
            'temp':'temp_air',
            'wind_spd':'wind_speed',
            'precipitable_water':'precipitable_water'
        }, inplace=True)
        sw_df = df[['temp_air','wind_speed','precipitable_water']]
        sw_df.to_csv('weatherbit_data.csv')
    else:
        sw_df = pd.read_csv(si['weatherbit_file'], index_col=0, parse_dates=True)

    # Phase B
    # Establish PVLIB Parameters, Modelchain, and Mount
    # Temperature model
    # Constant? Not sure if this will change
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
    # Copy Ideal Angles over to Stow Angles
    tracker_angles_2 = tracker_angles_1.copy()
    # Get Stow Conditions Angles
    tracker_angles_2 = run_stow_conditions(pd.concat([tracker_angles_2, sw_df]))
    # Recalculate AOI for Stow Angles
    # need to read and reference the paper behind this function
    # need to rename to remove poa as not part of below function
    tracker_angles_2 = recalculate_aoi_and_poa(tracker_angles_2,
                        si['axis_tilt'], si['axis_azimuth'])

    # Phase D
    # Build Weather Data and Estimate Power Output
    # Build weather data using different tracker angles to get POA
    results = []
    for i in [tracker_angles_1, tracker_angles_2]:
        wd = build_weather_data(w_df, i, solar_position, si['gcr'],
            si['axis_height'], si['pitch'], si['temperature_model_parameters'], si['module_unit_mass'])
        mc.run_model_from_poa(wd)
        ac = mc.results.ac / 1000
        dc = mc.results.dc['p_mp'] / 1000
        results.append([ac, dc])

    # Phase E
    # Analyze and Plot Results
    # Summarize results
    # print('Total energy output DC (kWh) - True Tracking:', dc.sum())
    # print('Total energy output DC (kWh) - Sample Wind Stow:', dc_v2.sum())
    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    # # True Tracking
    # dc.plot(ax=axes[0],
    #     title='Hourly Energy Output (kW) - True Tracking',
    #     ylabel='Power (kW)')
    # axes[0].set_xlabel('')
    # # Tracker Stall Example
    # dc_v2.plot(ax=axes[1],
    #     title='Hourly Energy Output (kW) - Sample Wind Stow',
    #     ylabel='Power (kW)')
    # axes[1].set_xlabel('Time')
    # plt.tight_layout()
    # plt.show()