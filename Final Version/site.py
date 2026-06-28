# Imports
import pvlib as pv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import RangeSlider
from sites import site_a
from datetime import time

class Site:
    def __init__ (self, name):
        self.name = name

    def import_stow_weather_data(self, file_path):
        self.selected_columns = [
            'Timestamp (UTC)',
            'Wind Speed (mph)',
            'Wind Gust Speed (mph)',
            'Wind Dir (Deg)',
            'Snowfall Rate (in/hr)',
            'Weather Code'
        ]
        # Code to import weather data from a file
        sw_df = pd.read_csv(file_path
                            , usecols=self.selected_columns
                            , parse_dates=['Timestamp (UTC)'])
        sw_df.set_index('Timestamp (UTC)', inplace=True)
        sw_df.rename(columns={
            'Wind Speed (mph)': 'Wind Speed (m/s)',
            'Wind Gust Speed (mph)': 'Wind Gust Speed (m/s)',
            'Snowfall Rate (in/hr)': 'Snowfall Rate (mm/hr)'
        }, inplace=True)

        sw_df['Wind Speed (m/s)'] = sw_df['Wind Speed (m/s)'] * 0.44704
        sw_df['Wind Gust Speed (m/s)'] = sw_df['Wind Gust Speed (m/s)'] * 0.44704
        sw_df['Snowfall Rate (mm/hr)'] = sw_df['Snowfall Rate (mm/hr)'] * 25.4
        self.sw_df = sw_df

    def build_array(self, param_dict):
        # Code to build an array of weather data
        # Temperature model
        temp_params = pv.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
        # PV module and inverter models (use realistic specs)
        cec_module_db = pv.pvsystem.retrieve_sam('cecmod')
        module_parameters = cec_module_db[param_dict['module_name']]
        # ensure that correct spectral correction is applied
        module_parameters['Technology'] = 'CdTe'
        cec_inverter_db = pv.pvsystem.retrieve_sam('cecinverter')
        inverter_parameters = cec_inverter_db[param_dict['inverter_name']]
        # Establish PVlib Location Object
        self.location = pv.location.Location(
            latitude=param_dict['latitude'],
            longitude=param_dict['longitude'],
            tz=param_dict['tz'],
            altitude=param_dict['altitude'],
            name=self.name
        )
        # Mount
        self.mount = pv.pvsystem.SingleAxisTrackerMount(
            axis_tilt=param_dict['axis_tilt'],
            axis_azimuth=param_dict['axis_azimuth'],
            max_angle=param_dict['max_angle'],
            backtrack=param_dict['backtrack']
        )
        # Array
        self.array = pv.pvsystem.Array(
            mount=self.mount,
            module_parameters=module_parameters,
            temperature_model_parameters=param_dict['temperature_model_parameters'],
            modules_per_string=param_dict['modules_per_string'],
            strings=param_dict['strings_per_inverter']
        )
        # System
        self.system = pv.pvsystem.PVSystem(
            arrays=[self.array], inverter_parameters=inverter_parameters
        )
        # Model Chain
        self.modelchain = pv.modelchain.ModelChain(
            self.system,
            self.location,
            ac_model='sandia',
            aoi_model='physical'
        )

    def get_ideal_tracker_angles(self):
        # Solar Position based off location and times
        solar_position = self.location.get_solarposition(self.sw_df.index)
        # Get Ideal Tracker Angles
        ideal_angle = self.mount.get_orientation(
            solar_position['apparent_zenith'],
            solar_position['azimuth'])
        ideal_angle['tracker_theta'] = ideal_angle['tracker_theta'].ffill()
        return ideal_angle['tracker_theta']
    
    def run_stow_conditions(self, df, relaxation_factor, max_angle):
        # Code to run stow conditions based on weather data
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

    def process_stow_conditions(self):
        # Code to run stow conditions based on weather data
        '''
        Adjusts tracker angles based on stow conditions
        '''
        df = self.sw_df.copy()
        df = df.assign(
            trigger=pd.Series(index=self.sw_df.index, dtype='object'),
            stow_angle=np.nan,
            stow_setpoint=np.nan,
            tracker_theta=self.get_ideal_tracker_angles()
        )
        df['td'] = pd.to_datetime(df.index)
        df['time_delta'] = (df['td']-df['td'].shift())
        relaxation_factor = 0; max_angle = 52
        # Set Initial Tracker Angle and Setpoint
        df.at[df.index[0], 'stow_setpoint'] = df.at[df.index[0], 'tracker_theta']
        df.at[df.index[0], 'stow_angle'] = df.at[df.index[0], 'tracker_theta']

        stow_conditions = self.run_stow_conditions(df, relaxation_factor, max_angle)
        return stow_conditions

    def save_results(self, output_path):
        # Code to save results to a file
        pass

    def combine_tracker_positions(self, fp1, fp2):
        stow_df = pd.read_csv(fp1, parse_dates=['Timestamp (UTC)']).set_index('Timestamp (UTC)')
        tracker_df = pd.read_csv(fp2, parse_dates=['Site Time']).set_index('Site Time')

        # stow timestamps are UTC, convert to local site time
        stow_df.index = (
            pd.to_datetime(stow_df.index, errors="coerce")
            .tz_localize("UTC")
            .tz_convert(site_a["tz"])
            .tz_localize(None)
        )
        
        stow_df.index = pd.to_datetime(stow_df.index, errors='coerce')
        tracker_df.index = pd.to_datetime(tracker_df.index, errors='coerce')

        stow_df = stow_df[stow_df.index.notna()]
        tracker_df = tracker_df[tracker_df.index.notna()]

        # Alter rest angle of real tracker
        # Build mask:
        # 12:00 AM -> 7:00 AM (inclusive) OR 4:15 PM -> 11:59:59 PM
        t = tracker_df.index.time
        mask = (t <= time(7, 0)) | (t >= time(16, 15))

        # Set pos columns to 0 during those hours
        tracker_df.loc[mask, ["pos1", "pos2", "pos3"]] = 0

        combined_df = pd.merge(
            stow_df,
            tracker_df,
            left_index=True,
            right_index=True,
            how='inner'
        ).sort_index()

        combined_df.index.name = 'timestamp'
        return combined_df

    def runall():

if __name__ == "__main__":
    x = Site("Site A")
    x.build_array(site_a)
    x.import_stow_weather_data("Final Version/WeatherBit Data/Site A.csv")
    x.get_ideal_tracker_angles()
    stow_conditions = x.process_stow_conditions()
    # print(stow_conditions.head(50))
    stow_conditions.to_csv("Final Version/Output/site_a_stow_conditions.csv")
    combined = x.combine_tracker_positions("Final Version/Output/site_a_stow_conditions.csv",
                                "Final Version/Nexamp Data/Site A.csv")
    combined.to_csv("Final Version/Output/site_a_combined.csv")
    # x.plot_scatter_with_datetime_slider(
    #     file_path="Final Version/Output/site_a_combined.csv",
    #     datetime_col="timestamp",
    #     y_col="stow_angle",
    #     usecols=["timestamp", "stow_angle", "tracker_theta", "pos1", "pos2", "pos3"],
    #     datetime_format="%Y-%m-%d %H:%M:%S",
    #     initial_window_days=7)