import pandas as pd
import numpy as np
import pvlib
import matplotlib.pyplot as plt
from site_info import site_info as si

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
    for idx, row in df.iterrows():
        # Stow Conditions
        # Storm
        # Hail
        # Wind
        if row['wind_speed'] > 10 or row['wind_gust_spd'] > 20:
            if row['tracker_theta'] < 0:
                df.at[idx, 'tracker_theta'] = -40
            else:
                df.at[idx, 'tracker_theta'] = 40
        # Snow
        # Flood

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
    w_df = pd.read_csv('weather_data.csv', index_col=0, parse_dates=True)
    # Import Stow Weather Data from Weatherbit
    sw_df = pd.read_csv('stow_weather_data.csv', index_col=0, parse_dates=True)

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
    wd_1 = build_weather_data(w_df, tracker_angles_1, solar_position, si['gcr'],
        si['axis_height'], si['pitch'], si['temperature_model_parameters'], si['module_unit_mass'])
    wd_2 = build_weather_data(w_df, tracker_angles_2, solar_position, si['gcr'],
        si['axis_height'], si['pitch'], si['temperature_model_parameters'], si['module_unit_mass'])
    # Estimate Power Output for Ideal Angles
    mc.run_model_from_poa(wd_1)
    ac = mc.results.ac / 1000
    dc = mc.results.dc['p_mp'] / 1000
    # Estimate Power Output for Stow Angles
    mc.run_model_from_poa(wd_2)
    ac_v2 = mc.results.ac / 1000
    dc_v2 = mc.results.dc['p_mp'] / 1000

    # Phase E
    # Analyze and Plot Results
    # Summarize results
    print('Total energy output DC (kWh) - True Tracking:', dc.sum())
    print('Total energy output DC (kWh) - Sample Wind Stow:', dc_v2.sum())
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    # True Tracking
    dc.plot(ax=axes[0],
        title='Hourly Energy Output (kW) - True Tracking',
        ylabel='Power (kW)')
    axes[0].set_xlabel('')
    # Tracker Stall Example
    dc_v2.plot(ax=axes[1],
        title='Hourly Energy Output (kW) - Sample Wind Stow',
        ylabel='Power (kW)')
    axes[1].set_xlabel('Time')
    plt.tight_layout()
    plt.show()
