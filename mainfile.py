import pvlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Metadata for site and system (edit for your location/system)
latitude, longitude = 39.74, -104.985 # Denver, CO example
tz = 'MST'
system_capacity = 4700 # kW (for 4.7 MW system)
tilt, azimuth = 0, 180 # Initial orientation

# Create time range for modeling (one day example)
times = pd.date_range('2025-06-21', '2025-06-22', freq='1h', tz=tz)

# Solar position and tracking
solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
tracking = pvlib.tracking.singleaxis(
    apparent_zenith=solpos['apparent_zenith'],
    apparent_azimuth=solpos['azimuth'],
    axis_tilt=0, axis_azimuth=90, max_angle=60, backtrack=True, 
    gcr=0.3
)

# Weather data stub (replace with measured data)
dni = 800  # Direct Normal Irradiance W/m2
ghi = 500  # Global Horizontal Irradiance W/m2
dhi = 100  # Diffuse Horizontal Irradiance W/m2

weather = pd.DataFrame({
    'dni': dni, 'ghi': ghi, 'dhi': dhi
}, index=times)

# Surface irradiance with tracker rotation
poa_irradiance = pvlib.irradiance.get_total_irradiance(
    surface_tilt=tracking['tilt'], 
    surface_azimuth=tracking['azimuth'], 
    solar_zenith=solpos['apparent_zenith'],
    solar_azimuth=solpos['azimuth'],
    dni=weather['dni'],
    ghi=weather['ghi'],
    dhi=weather['dhi']
)

# PV module and inverter models (use realistic specs)
sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
cec_inverters = pvlib.pvsystem.retrieve_sam('CECInverter')
inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

# Temperature model
temp_cell = pvlib.temperature.sapm_cell(
    poa_irradiance['poa_global'],
    temp_air=25, wind_speed=1
)

# PV system instance
system = pvlib.pvsystem.PVSystem(
    surface_tilt=tilt, surface_azimuth=azimuth, 
    module_parameters=module, inverter_parameters=inverter
)

# DC output
dc = system.sapm(poa_irradiance, temp_cell)

# AC output
ac = system.inverter(dc['v_mp'], dc['p_mp'])

# Summarize results
print('Total energy output (kWh):', ac.sum()/1000)
ac.plot(title='Hourly Energy Output (W)', ylabel='Power (W)')
plt.show()
