site_info = \
{
    'latitude' : 42.38,
    'longitude' : -71.099,
    'tz' : 'EST',
    'altitude' : 100,
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
    'module_unit_mass' : 12 / 0.72  # kg/m^2, taken from datasheet values
}