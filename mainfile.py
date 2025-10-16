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
times = pd.date_range('2025-06-21', '2025-06-22', freq='1h', tz=tz)
# System Parameters
axis_tilt = 0
axis_azimuth = 0
max_angle = 60
backtrack = True
modules_per_string = 20
strings_per_inverter = 1
# Temperature model
temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
# PV module and inverter models (use realistic specs)
sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
cec_inverters = pvlib.pvsystem.retrieve_sam('CECInverter')
inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
solar_position = location.get_solarposition(times)
api_key = 'Qwu9Ny75sGchX3wCrjcgFX7PePxJSMOt0xUgTeXC'
email = "wsbzan@gmail.com"
keys = ['ghi','dni','dhi','temp_air','wind_speed','albedo','precipitable_water']
psm3, psm3_metadata = pvlib.iotools.get_psm3(latitude, longitude, api_key,
                                            email, interval = 15, names=2020,
                                            map_variables=True, leap_day=True,
                                            attributes=keys)
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
    solar_position['azimuth']
)
print(tracker_angles.head(5))
# Array
array = pvlib.pvsystem.Array(
    mount=mount,
    module_parameters=module,
    temperature_model_parameters = temp_params,
    modules_per_string = modules_per_string,
    strings = strings_per_inverter
)
# System
system = pvlib.pvsystem.PVSystem(
    arrays=[array], inverter_parameters=inverter
)
# Model Chain
modelchain = pvlib.modelchain.ModelChain(
    system,
    location,
    ac_model = 'sandia',
    aoi_model='physical'
)
modelchain.run_model_from_effective_irradiance(data=data)
ac = modelchain.results.ac / 1000
dc = modelchain.results.dc['p_mp'] / 1000

# Summarize results
print('Total energy output (kWh):', ac.sum())
ac.plot(title='Hourly Energy Output (W)', ylabel='Power (W)')
plt.show()
