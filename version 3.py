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
    # Create Location Object
    location = pvlib.location.Location(
        latitude=si['latitude'],
        longitude=si['longitude'],
        tz=si['tz'],
        altitude=si['altitude'],
        name="Example Site"
    )
    # Establish time range
    times = pd.date_range('2024-01-01', '2024-01-31', freq='15m', tz=si['tz'])
    # Get Solar Position
    solar_position = location.get_solarposition(times)
    # PVLIB Parameters
    # Temperature model
    temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
    # PV module and inverter models (use realistic specs)
    cec_module_db = pvlib.pvsystem.retrieve_sam('cecmod')
    module_parameters = cec_module_db['First_Solar__Inc__FS_4117_3']
    # ensure that correct spectral correction is applied
    module_parameters['Technology'] = 'CdTe'
    cec_inverter_db = pvlib.pvsystem.retrieve_sam('cecinverter')
    inverter_parameters = cec_inverter_db['TMEIC__PVL_L1833GRM']

    # psm3 Weather Data
    api_key = 'Qwu9Ny75sGchX3wCrjcgFX7PePxJSMOt0xUgTeXC'
    email = "wsbzan@gmail.com"
    keys = ['ghi','dni','dhi','temp_air','wind_speed','albedo','precipitable_water']
    try:
        psm3, psm3_metadata = pvlib.iotools.get_psm3(si['latitude'], si['longitude'], api_key,
                                                    email, interval = 15, names=2015,
                                                    map_variables=True, leap_day=True,
                                                    attributes=keys)
    except Exception as e:
        print("Error retrieving PSM3 data:", e)
    psm3 = psm3.loc[times]

    # Access Functions
    mc, mount = build_site(si['axis_tilt'], si['axis_azimuth'], si['max_angle'], si['backtrack'],
        module_parameters,si['temperature_model_parameters'], si['modules_per_string'],
        si['strings_per_inverter'], inverter_parameters, location)

    # Import Stow Weather Data Sample
    stow_weather_data = pd.read_csv('sample data.csv')
    stow_weather_data.index = pd.to_datetime(stow_weather_data['timestamp_local'])

    # Ideal Angles
    tracker_angles_1 = mount.get_orientation(
        solar_position['apparent_zenith'],
        solar_position['azimuth'])
    # Stow Angles
    tracker_angles_2 = tracker_angles_1.copy()
    # Adjust Angle Based on Stow Conditions
    tracker_angles_2 = run_stow_conditions(pd.concat([tracker_angles_2, stow_weather_data]))

    # need to read and reference the paper behind this function
    tracker_angles_2 = recalculate_aoi_and_poa(tracker_angles_2,
                        si['axis_tilt'], si['axis_azimuth'])

    # Build weather data using different tracker angles to get POA
    wd_1 = build_weather_data(psm3, tracker_angles_1, solar_position, si['gcr'],
        si['axis_height'], si['pitch'], si['temperature_model_parameters'], si['module_unit_mass'])
    wd_2 = build_weather_data(psm3, tracker_angles_2, solar_position, si['gcr'],
        si['axis_height'], si['pitch'], si['temperature_model_parameters'], si['module_unit_mass'])
    mc.run_model_from_poa(wd_1)
    ac = mc.results.ac / 1000
    dc = mc.results.dc['p_mp'] / 1000

    mc.run_model_from_poa(wd_2)
    ac_v2 = mc.results.ac / 1000
    dc_v2 = mc.results.dc['p_mp'] / 1000

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
