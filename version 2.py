import pandas as pd
import numpy as np
import pvlib
import matplotlib.pyplot as plt

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
        'precipitable_water': psm3['precipitable_water'],  # for the spectral model
    })

if __name__ == '__main__':
    # Input Parameters
    latitude, longitude = 39.74, -104.985 # Denver, CO example
    tz = 'MST'
    altitude = 1000
    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        tz=tz,
        altitude=altitude,
        name="Example Site"
    )
    # Create time range
    times = pd.date_range('2020-06-21', '2020-06-25', freq='1h', tz=tz)
    # System Parameters
    axis_tilt = 0
    axis_azimuth = 180
    max_angle = 60
    backtrack = True
    modules_per_string = 200
    strings_per_inverter = 1
    # Temperature model
    temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
    gcr=0.001 #single row, gcr -> 0
    axis_height = 1 # meter
    pitch = 5 # m
    # default Faiman model parameters:
    temperature_model_parameters = dict(u0=25.0, u1=6.84)
    module_unit_mass = 12 / 0.72  # kg/m^2, taken from datasheet values
    # PV module and inverter models (use realistic specs)
    cec_module_db = pvlib.pvsystem.retrieve_sam('cecmod')
    module_parameters = cec_module_db['First_Solar__Inc__FS_4117_3']
    # ensure that correct spectral correction is applied
    module_parameters['Technology'] = 'CdTe'
    cec_inverter_db = pvlib.pvsystem.retrieve_sam('cecinverter')
    inverter_parameters = cec_inverter_db['TMEIC__PVL_L1833GRM']
    solar_position = location.get_solarposition(times)
    api_key = 'Qwu9Ny75sGchX3wCrjcgFX7PePxJSMOt0xUgTeXC'
    email = "wsbzan@gmail.com"
    keys = ['ghi','dni','dhi','temp_air','wind_speed','albedo','precipitable_water']
    psm3, psm3_metadata = pvlib.iotools.get_psm3(latitude, longitude, api_key,
                                                email, interval = 15, names=2020,
                                                map_variables=True, leap_day=True,
                                                attributes=keys)
    psm3 = psm3.loc[times]

    # Access Functions
    mc, mount = build_site(axis_tilt, axis_azimuth, max_angle, backtrack,
        module_parameters,temperature_model_parameters, modules_per_string,
        strings_per_inverter, inverter_parameters, location)

    # Adjust Tracker Angle
    # Ideal Angle
    tracker_angles_1 = mount.get_orientation(
        solar_position['apparent_zenith'],
        solar_position['azimuth'])
    # Example Adjusted Angle
    # Tracker Stall at Noon on First Day to -25
    tracker_angles_2 = tracker_angles_1.copy()
    tracker_angles_2['tracker_theta'].iloc[12:95]=-55
    # need to read and reference the paper behind this function
    surface = pvlib.tracking.calc_surface_orientation(
            tracker_angles_2['tracker_theta'], axis_tilt, axis_azimuth)
    surface_tilt = surface['surface_tilt']
    surface_azimuth = surface['surface_azimuth']
    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth,
            solar_position['apparent_zenith'], solar_position['azimuth'])
    tracker_angles_2['aoi'] = aoi
    tracker_angles_2['surface_tilt'] = surface_tilt
    tracker_angles_2['surface_azimuth'] = surface_azimuth

    # Build weather data using different tracker angles to get POA
    wd_1 = build_weather_data(psm3, tracker_angles_1, solar_position, gcr,
        axis_height, pitch, temperature_model_parameters, module_unit_mass)

    wd_2 = build_weather_data(psm3, tracker_angles_2, solar_position, gcr,
        axis_height, pitch, temperature_model_parameters, module_unit_mass)

    mc.run_model_from_poa(wd_1)
    ac = mc.results.ac / 1000
    dc = mc.results.dc['p_mp'] / 1000

    mc.run_model_from_poa(wd_2)
    ac_v2 = mc.results.ac / 1000
    dc_v2 = mc.results.dc['p_mp'] / 1000

    # Summarize results
    print('Total energy output DC (kWh) - True Tracking:', dc.sum())
    print('Total energy output DC (kWh) - Tracker Stall:', dc_v2.sum())
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    # True Tracking
    dc.plot(ax=axes[0],
        title='Hourly Energy Output (kW) - True Tracking',
        ylabel='Power (kW)')
    axes[0].set_xlabel('')
    # Tracker Stall Example
    dc_v2.plot(ax=axes[1],
        title='Hourly Energy Output (kW) - Tracker Stall at -55 Deg',
        ylabel='Power (kW)')
    axes[1].set_xlabel('Time')
    plt.tight_layout()
    plt.show()
