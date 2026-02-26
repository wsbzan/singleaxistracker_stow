import pvlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Metadata for site and system (edit for your location/system)
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
print(psm3.head(5))
# Mount
mount = pvlib.pvsystem.SingleAxisTrackerMount(
    axis_tilt=axis_tilt,
    axis_azimuth=axis_azimuth,
    max_angle=max_angle,
    backtrack=backtrack
)
tracker_angles = mount.get_orientation(
    solar_position['apparent_zenith'],
    solar_position['azimuth'])
tracker_angles.to_csv("tracker_angles.csv",index=True)
tracker_angles_v2 = tracker_angles.copy()
tracker_angles_v2['tracker_theta'].iloc[12:35]=-25
# need to read and reference the paper behind this function
surface = pvlib.tracking.calc_surface_orientation(tracker_angles_v2['tracker_theta'], axis_tilt, axis_azimuth)
surface_tilt = surface['surface_tilt']
surface_azimuth = surface['surface_azimuth']
aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth,
                     solar_position['apparent_zenith'],
                     solar_position['azimuth'])
tracker_angles_v2['aoi'] = aoi
tracker_angles_v2['surface_tilt'] = surface_tilt
tracker_angles_v2['surface_azimuth'] = surface_azimuth
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
# Copied from pvlib example
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

weather_inputs = pd.DataFrame({
    'poa_global': averaged_irradiance['poa_global'],
    'poa_direct': averaged_irradiance['poa_direct'],
    'poa_diffuse': averaged_irradiance['poa_diffuse'],
    'cell_temperature': cell_temperature,
    'precipitable_water': psm3['precipitable_water'],  # for the spectral model
})
modelchain.run_model_from_poa(weather_inputs)
ac = modelchain.results.ac / 1000
dc = modelchain.results.dc['p_mp'] / 1000
weather_inputs.to_csv("weather_data.csv",index=True)
ac.to_csv("ac.csv",index=True)
dc.to_csv("dc.csv",index=True)
# adjusted angle
averaged_irradiance_v2 = pvlib.bifacial.infinite_sheds.get_irradiance_poa(
    tracker_angles_v2['surface_tilt'], tracker_angles_v2['surface_azimuth'],
    solar_position['apparent_zenith'], solar_position['azimuth'],
    gcr, axis_height, pitch,
    psm3['ghi'], psm3['dhi'], psm3['dni'], psm3['albedo'],
    model='haydavies', dni_extra=dni_extra,
)
cell_temperature_steady_state_v2 = pvlib.temperature.faiman(
    poa_global=averaged_irradiance_v2['poa_global'],
    temp_air=psm3['temp_air'],
    wind_speed=psm3['wind_speed'],
    **temperature_model_parameters,
)
cell_temperature_v2 = pvlib.temperature.prilliman(
    cell_temperature_steady_state_v2,
    psm3['wind_speed'],
    unit_mass=module_unit_mass
)

weather_inputs_v2 = pd.DataFrame({
    'poa_global': averaged_irradiance_v2['poa_global'],
    'poa_direct': averaged_irradiance_v2['poa_direct'],
    'poa_diffuse': averaged_irradiance_v2['poa_diffuse'],
    'cell_temperature': cell_temperature_v2,
    'precipitable_water': psm3['precipitable_water'],  # for the spectral model
})
modelchain.run_model_from_poa(weather_inputs_v2)
ac_v2 = modelchain.results.ac / 1000
dc_v2 = modelchain.results.dc['p_mp'] / 1000
# Summarize results
print('Total energy output DC (kWh):', dc.sum())
dc.plot(title='Hourly Energy Output (W)', ylabel='Power (W)')
plt.show()
# Summarize results
print('Total energy output DC (kWh):', dc_v2.sum())
dc_v2.plot(title='Hourly Energy Output (W)', ylabel='Power (W)')
plt.show()
