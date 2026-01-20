site_info = \
{
    # Site Name and Location
    'name' : 'Example Site',
    'latitude' : 42.38,
    'longitude' : -71.09,
    'tz' : 'EST',
    'altitude' : 100,
    # Date Range and Frequency
    'start' : '2024-01-01',
    'end' : '2024-01-31',
    'freq' : '15min',
    # Data Import
    'psm': False,
    'weaterbit' : False,
    'psm_file' : 'psm4_data.csv',
    'weatherbit_file' : 'weatherbit_data.csv',
    # System Parameters
    'axis_tilt' : 0,
    'axis_azimuth' : 180,
    'max_angle' : 60,
    'backtrack' : True,
    'modules_per_string' : 200,
    'strings_per_inverter' : 1,
    'gcr' : 0.001, #single row, gcr -> 0
    'axis_height' : 1, # meter
    'pitch' : 5, # m
    # default Faiman model parameters:
    'temperature_model_parameters' : dict(u0=25.0, u1=6.84),
    'module_unit_mass' : 12 / 0.72,  # kg/m^2, taken from datasheet values
    # Array, Inverter, and Module parameters
    'module_name' : 'First_Solar__Inc__FS_4117_3',
    'inverter__name' : 'TMEIC__PVL_L1833GRM'
}