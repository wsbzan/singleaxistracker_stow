# Imports
import os
import pvlib as pv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import RangeSlider
from datetime import time
from sites import sites as SITE_CONFIGS

# Default stow-algorithm feature toggles. Override per-run by passing a partial
# dict (merged over these defaults) to process_stow_conditions/runall/run_initial_site.
DEFAULT_STOW_TOGGLES = {
    'enable_storm': True,
    'enable_wind': True,
    'enable_snow': True,
    # zero_crossing=True: stay on the current side of flat (0 deg), never sweep through it.
    # zero_crossing=False: ignore the current side and point the backside into the wind instead.
    'storm_zero_crossing': False,
    'wind_zero_crossing': False,
    'snow_zero_crossing': False,
    # Preemptive triggers look ahead `preempt_steps` timestamps for a storm/snow
    # condition and, if found, stow early under a 'Preemptive <Condition>' label.
    'preempt_storm': False,
    'preempt_snow': False,
    'preempt_steps': 2,
}

class Site:
    def __init__ (self, name):
        self.name = name
        self.tz = 'America/New_York'

    def import_stow_weather_data(self, file_path):
        # WeatherBit exports are not fully consistent across sites (imperial vs metric).
        sw_df = pd.read_csv(file_path, parse_dates=['Timestamp (UTC)'])

        if 'Wind Speed (mph)' in sw_df.columns:
            sw_df['Wind Speed (m/s)'] = pd.to_numeric(sw_df['Wind Speed (mph)'], errors='coerce') * 0.44704
        elif 'Wind Speed (m/s)' in sw_df.columns:
            sw_df['Wind Speed (m/s)'] = pd.to_numeric(sw_df['Wind Speed (m/s)'], errors='coerce')
        else:
            raise ValueError(f"Missing wind speed column in {file_path}")

        if 'Wind Gust Speed (mph)' in sw_df.columns:
            sw_df['Wind Gust Speed (m/s)'] = pd.to_numeric(sw_df['Wind Gust Speed (mph)'], errors='coerce') * 0.44704
        elif 'Wind Gust Speed (m/s)' in sw_df.columns:
            sw_df['Wind Gust Speed (m/s)'] = pd.to_numeric(sw_df['Wind Gust Speed (m/s)'], errors='coerce')
        else:
            # Fallback if gust data is not provided.
            sw_df['Wind Gust Speed (m/s)'] = sw_df['Wind Speed (m/s)']

        if 'Snowfall Rate (in/hr)' in sw_df.columns:
            sw_df['Snowfall Rate (mm/hr)'] = pd.to_numeric(sw_df['Snowfall Rate (in/hr)'], errors='coerce') * 25.4
        elif 'Snowfall Rate (mm/hr)' in sw_df.columns:
            sw_df['Snowfall Rate (mm/hr)'] = pd.to_numeric(sw_df['Snowfall Rate (mm/hr)'], errors='coerce')
        else:
            sw_df['Snowfall Rate (mm/hr)'] = 0.0

        required_columns = ['Timestamp (UTC)', 'Wind Dir (Deg)', 'Weather Code']
        missing_columns = [c for c in required_columns if c not in sw_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {file_path}: {missing_columns}")

        sw_df['Wind Dir (Deg)'] = pd.to_numeric(sw_df['Wind Dir (Deg)'], errors='coerce')
        sw_df['Weather Code'] = pd.to_numeric(sw_df['Weather Code'], errors='coerce')

        self.selected_columns = [
            'Timestamp (UTC)',
            'Wind Speed (m/s)',
            'Wind Gust Speed (m/s)',
            'Wind Dir (Deg)',
            'Snowfall Rate (mm/hr)',
            'Weather Code',
        ]

        sw_df = sw_df[self.selected_columns].copy()
        sw_df.set_index('Timestamp (UTC)', inplace=True)
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
            backtrack=param_dict['backtrack'],
            gcr = param_dict['gcr']
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
        self.tz = param_dict['tz']

    def get_ideal_tracker_angles(self):
        # Solar Position based off location and times
        solar_position = self.location.get_solarposition(self.sw_df.index)
        # Get Ideal Tracker Angles
        ideal_angle = self.mount.get_orientation(
            solar_position['apparent_zenith'],
            solar_position['azimuth'])
        ideal_angle['tracker_theta'] = ideal_angle['tracker_theta'].ffill()
        return ideal_angle['tracker_theta']
    
    @staticmethod
    def _directional_setpoint(magnitude, zero_crossing, current_angle, wind_dir_deg, axis_azimuth):
        # Option a: stay on the current side of flat (0 deg) so the tracker never crosses through it.
        if zero_crossing:
            return -magnitude if current_angle < 0 else magnitude
        # Option b: backside into the wind. With axis_tilt=0, +theta faces (axis_azimuth+90)
        # and -theta faces (axis_azimuth-90); the front should face downwind (wind_dir+180),
        # so pick +theta when downwind falls on the (axis_azimuth+90) half of the circle.
        offset = (wind_dir_deg + 180 - axis_azimuth) % 360
        return magnitude if offset < 180 else -magnitude

    def run_stow_conditions(
        self, df, relaxation_factor, max_angle,
        enable_storm=True, enable_wind=True, enable_snow=True,
        storm_zero_crossing=True, wind_zero_crossing=True, snow_zero_crossing=False,
        preempt_storm=False, preempt_snow=False, preempt_steps=2
    ):
        # Use NumPy-backed arrays to avoid slow per-row pandas indexing on large files.
        wind_speed = df['Wind Speed (m/s)'].to_numpy()
        wind_gust = df['Wind Gust Speed (m/s)'].to_numpy()
        wind_dir = df['Wind Dir (Deg)'].to_numpy()
        snowfall = df['Snowfall Rate (mm/hr)'].to_numpy()
        weather_code = df['Weather Code'].to_numpy()
        time_delta_mins = (df['time_delta'].dt.total_seconds() / 60).to_numpy()
        trigger = df['trigger'].to_numpy()
        stow_angle = df['stow_angle'].to_numpy()
        stow_setpoint = df['stow_setpoint'].to_numpy()
        tracker_theta = df['tracker_theta'].to_numpy()

        storm_codes = {200, 201, 202, 230, 231, 232, 233, 511}
        snow_codes = {600, 601, 602, 610, 611, 612, 621, 622, 623}
        max_angle_change = 30
        wind_magnitude = 40
        axis_azimuth = self.mount.axis_azimuth

        def storm_condition(idx):
            return wind_speed[idx] > 25 or weather_code[idx] in storm_codes

        def snow_condition(idx):
            return snowfall[idx] > 0 or weather_code[idx] in snow_codes

        for i in range(len(df) - 1):
            next_i = i + 1
            has_trigger = not pd.isna(trigger[i])

            storm_now = enable_storm and storm_condition(i)
            wind_now = enable_wind and (wind_speed[i] > 11 or wind_gust[i] > 16)
            snow_now = enable_snow and snow_condition(i)

            if storm_now:
                if not has_trigger:
                    trigger[i] = 'Storm'
                    relaxation_factor = 40
                    stow_setpoint[next_i] = self._directional_setpoint(
                        max_angle, storm_zero_crossing, stow_angle[i], wind_dir[i], axis_azimuth)
            elif wind_now:
                if not has_trigger:
                    trigger[i] = 'Wind'
                    relaxation_factor = 20
                    stow_setpoint[next_i] = self._directional_setpoint(
                        wind_magnitude, wind_zero_crossing, stow_angle[i], wind_dir[i], axis_azimuth)
            elif snow_now:
                if not has_trigger:
                    trigger[i] = 'Snow'
                    relaxation_factor = 30
                    stow_setpoint[next_i] = self._directional_setpoint(
                        max_angle, snow_zero_crossing, stow_angle[i], wind_dir[i], axis_azimuth)
            else:
                # Priority #4: only consider a preemptive stow once no real trigger fired this step.
                preemptive_fired = False
                if preempt_storm and enable_storm:
                    for offset in range(1, preempt_steps + 1):
                        j = i + offset
                        if j < len(df) and storm_condition(j):
                            if not has_trigger:
                                trigger[i] = 'Preemptive Storm'
                                relaxation_factor = 40
                                stow_setpoint[next_i] = self._directional_setpoint(
                                    max_angle, storm_zero_crossing, stow_angle[i], wind_dir[j], axis_azimuth)
                            preemptive_fired = True
                            break
                if not preemptive_fired and preempt_snow and enable_snow:
                    for offset in range(1, preempt_steps + 1):
                        j = i + offset
                        if j < len(df) and snow_condition(j):
                            if not has_trigger:
                                trigger[i] = 'Preemptive Snow'
                                relaxation_factor = 30
                                stow_setpoint[next_i] = self._directional_setpoint(
                                    max_angle, snow_zero_crossing, stow_angle[i], wind_dir[j], axis_azimuth)
                            preemptive_fired = True
                            break
                if not preemptive_fired:
                    if relaxation_factor > 0:
                        trigger[i] = 'Relaxing'
                        relaxation_factor -= time_delta_mins[i]
                        if relaxation_factor > 0:
                            stow_setpoint[next_i] = stow_setpoint[i]
                        else:
                            stow_setpoint[next_i] = tracker_theta[next_i]
                    else:
                        stow_setpoint[next_i] = tracker_theta[next_i]
                        trigger[i] = 'Ideal Tracking'

            setpoint_angle = stow_setpoint[next_i]
            current_angle = stow_angle[i]
            angle_delta = setpoint_angle - current_angle

            if abs(angle_delta) > max_angle_change:
                if setpoint_angle < current_angle:
                    stow_angle[next_i] = np.clip(current_angle - max_angle_change, -max_angle, max_angle)
                else:
                    stow_angle[next_i] = np.clip(current_angle + max_angle_change, -max_angle, max_angle)
            else:
                stow_angle[next_i] = setpoint_angle

        return df

    def process_stow_conditions(self, stow_toggles=None):
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
        relaxation_factor = 0; max_angle = self.mount.max_angle
        # Set Initial Tracker Angle and Setpoint
        df.at[df.index[0], 'stow_setpoint'] = df.at[df.index[0], 'tracker_theta']
        df.at[df.index[0], 'stow_angle'] = df.at[df.index[0], 'tracker_theta']

        toggles = {**DEFAULT_STOW_TOGGLES, **(stow_toggles or {})}
        stow_conditions = self.run_stow_conditions(df, relaxation_factor, max_angle, **toggles)
        return stow_conditions
    
    def save_results(self, output_path):
        # Code to save results to a file
        pass

    def combine_tracker_positions(self, fp1, fp2):
        stow_df = pd.read_csv(fp1, parse_dates=['Timestamp (UTC)']).set_index('Timestamp (UTC)')
        tracker_df = pd.read_csv(fp2, parse_dates=['Site Time']).set_index('Site Time')
        site_tz = getattr(self, 'tz')
        # Normalize both tracker and stow timestamps to UTC so the merged output has
        # a single, unambiguous timeline across DST transitions.
        stow_df.index = pd.to_datetime(stow_df.index, errors='coerce', utc=True)

        tracker_df = tracker_df[tracker_df.index.notna()].sort_index()
        tracker_idx = pd.to_datetime(tracker_df.index, errors='coerce')
        tracker_df.index = tracker_idx.tz_localize(
            site_tz,
            ambiguous=~tracker_idx.duplicated(keep='first'),
            nonexistent='shift_forward'
        ).tz_convert('UTC')

        stow_df = stow_df[stow_df.index.notna()]
        tracker_df = tracker_df[tracker_df.index.notna()]

        # # Alter rest angle of real tracker
        # # Build mask:
        # # 12:00 AM -> 7:00 AM (inclusive) OR 4:15 PM -> 11:59:59 PM
        # t = tracker_df.index.time
        # mask = (t <= time(7, 0)) | (t >= time(16, 15))

        # # Set pos columns to 0 during those hours
        # tracker_df.loc[mask, ["pos1", "pos2", "pos3"]] = 0

        combined_df = pd.merge(
            stow_df,
            tracker_df,
            left_index=True,
            right_index=True,
            how='inner'
        ).sort_index()

        combined_df.index.name = 'timestamp'

        # Ensure td always matches timestamp timezone basis in combined output.
        if 'td' in combined_df.columns:
            combined_df['td'] = combined_df.index

        return combined_df

    def plot_combined_scatter(self, combined_df, output_path):
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.scatter(combined_df.index, combined_df['stow_angle'], s=8, alpha=0.6, label='stow_angle')

        if 'tracker_theta' in combined_df.columns:
            ax.scatter(combined_df.index, combined_df['tracker_theta'], s=8, alpha=0.6, label='tracker_theta')
        if 'pos1' in combined_df.columns:
            ax.scatter(combined_df.index, combined_df['pos1'], s=8, alpha=0.6, label='pos1')
        if 'pos2' in combined_df.columns:
            ax.scatter(combined_df.index, combined_df['pos2'], s=8, alpha=0.6, label='pos2')
        if 'pos3' in combined_df.columns:
            ax.scatter(combined_df.index, combined_df['pos3'], s=8, alpha=0.6, label='pos3')

        ax.set_title(f'{self.name} Combined Tracker Scatter')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Angle (deg)')
        ax.grid(alpha=0.3)
        ax.legend(loc='best')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def runall(stow_toggles=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weather_dir = os.path.join(base_dir, 'WeatherBit Data')
        nexamp_dir = os.path.join(base_dir, 'Nexamp Data')
        output_dir = os.path.join(base_dir, 'Output')
        os.makedirs(output_dir, exist_ok=True)

        base_params = {
            'tz': 'America/New_York',
            'altitude': 135,
            'axis_tilt': 0,
            'axis_azimuth': 180,
            'max_angle': 52,
            'backtrack': True,
            'modules_per_string': 200,
            'strings_per_inverter': 1,
            'temperature_model_parameters': dict(u0=25.0, u1=6.84),
            'module_name': 'First_Solar__Inc__FS_4117_3',
            'inverter_name': 'TMEIC__PVL_L1833GRM',
        }

        site_files = {
            'site_a': 'Site A.csv',
            'site_b': 'Site B.csv',
            'site_c': 'Site C.csv',
            'site_d': 'Site D.csv',
            'site_e': 'Site E.csv',
        }

        for site_key, csv_name in site_files.items():
            print(site_key)
            weather_path = os.path.join(weather_dir, csv_name)
            nexamp_path = os.path.join(nexamp_dir, csv_name)

            if not os.path.exists(weather_path):
                print(f'Skipping {site_key}: missing weather file {weather_path}')
                continue
            if not os.path.exists(nexamp_path):
                print(f'Skipping {site_key}: missing tracker file {nexamp_path}')
                continue

            config_from_sites = SITE_CONFIGS.get(site_key, {}) if isinstance(SITE_CONFIGS, dict) else {}
            params = {
                **base_params,
                **config_from_sites,
            }

            site_name = site_key.replace('_', ' ').title()
            site_obj = Site(site_name)
            site_obj.build_array(params)
            site_obj.import_stow_weather_data(weather_path)

            stow_conditions = site_obj.process_stow_conditions(stow_toggles)
            stow_out = os.path.join(output_dir, f'{site_key}_stow_conditions.csv')
            stow_conditions.to_csv(stow_out)

            combined = site_obj.combine_tracker_positions(stow_out, nexamp_path)
            combined_out = os.path.join(output_dir, f'{site_key}_combined.csv')
            combined.to_csv(combined_out)

            scatter_out = os.path.join(output_dir, f'{site_key}_scatter.png')
            site_obj.plot_combined_scatter(combined, scatter_out)

            print(f'Finished {site_key}: {combined_out}, {scatter_out}')

    @staticmethod
    def run_initial_site(site_key='site_example', csv_name='example site dataset.csv', stow_toggles=None, output_name=None):
        '''
        Runs the stow-condition algorithm on the initial/example dataset
        (brought over from version 4.py / site_info.py) instead of a real
        validation site. There is no real tracker ground truth for this
        dataset, so only the stow-conditions timeline is produced.

        output_name: filename prefix for the output CSV (defaults to site_key).
        Pass a distinct tag per stow_toggles combination to keep multiple runs'
        outputs side-by-side for later plotting/metrics comparison.
        '''
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weather_path = os.path.join(base_dir, csv_name)
        output_dir = os.path.join(base_dir, 'Output')
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(weather_path):
            print(f'Skipping {site_key}: missing data file {weather_path}')
            return None

        params = SITE_CONFIGS[site_key]
        site_obj = Site(site_key.replace('_', ' ').title())
        site_obj.build_array(params)
        site_obj.import_stow_weather_data(weather_path)

        stow_conditions = site_obj.process_stow_conditions(stow_toggles)
        stow_out = os.path.join(output_dir, f'{output_name or site_key}_stow_conditions.csv')
        stow_conditions.to_csv(stow_out)

        print(f'Finished {site_key}: {stow_out}')
        print(stow_conditions['trigger'].value_counts())
        return stow_conditions

    @staticmethod
    def mae(fp):
        df = pd.read_csv(fp, parse_dates=['timestamp']).set_index('timestamp')
        df['avg_pos_angle'] = df[['pos1', 'pos2', 'pos3']].mean(axis=1)

        # Keep only timestamps where stow_angle is within 20 degrees of avg_pos_angle.
        within_20 = np.abs(df['stow_angle'] - df['avg_pos_angle']) <= 20
        outside_20 = np.abs(df['stow_angle'] - df['avg_pos_angle']) > 20
        filtered_df_w = df.loc[within_20, ['avg_pos_angle', 'stow_angle']].dropna()
        filtered_df_o = df.loc[outside_20, ['avg_pos_angle', 'stow_angle']].dropna()
        active_stow = df['trigger']!= "Ideal Tracking"
        ideal_tracking = df['trigger']== "Ideal Tracking"
        filtered_df_as = df.loc[active_stow, ['avg_pos_angle', 'stow_angle']].dropna()
        filtered_df_it = df.loc[ideal_tracking, ['avg_pos_angle', 'stow_angle']].dropna()

        mae_value_w = np.mean(np.abs(filtered_df_w['avg_pos_angle'] - filtered_df_w['stow_angle']))
        mae_value_o = np.mean(np.abs(filtered_df_o['avg_pos_angle'] - filtered_df_o['stow_angle']))
        percent_w = (len(filtered_df_w) / len(df)) * 100
        percent_o = (len(filtered_df_o) / len(df)) * 100

        mae_value_as = np.mean(np.abs(filtered_df_as['avg_pos_angle'] - filtered_df_as['stow_angle']))
        mae_value_it = np.mean(np.abs(filtered_df_it['avg_pos_angle'] - filtered_df_it['stow_angle']))
        percent_as = (len(filtered_df_as) / len(df)) * 100
        percent_it = (len(filtered_df_it) / len(df)) * 100

        print(len(df))
        print(len(filtered_df_w))
        print(len(filtered_df_o))
        print(len(filtered_df_as))
        print(len(filtered_df_it))

        return {
            'mae_within_20': mae_value_w,
            'mae_outside_20': mae_value_o,
            'percent_within_20': percent_w,
            'percent_outside_20': percent_o,
            'mae_active_stow': mae_value_as,
            'mae_ideal_tracking': mae_value_it,
            'percent_active_stow': percent_as,
            'percent_ideal_tracking': percent_it
        }

    @staticmethod
    def stow_metrics(fp, weather_fp, year, loc, axis_tilt=0, axis_azimuth=180):
        df = pd.read_csv(fp, parse_dates=['timestamp']).set_index('timestamp')
        df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
        # Convert to local time
        df.index = df.index.tz_convert(loc.tz)
        df = df[df.index.notna()]
        # Restrict metrics calculations to the 2024 calendar year.
        df = df[df.index.year == year]
        df['avg_pos_angle'] = df[['pos1', 'pos2', 'pos3']].mean(axis=1)
        weather_df = pd.read_csv(weather_fp)
        weather_df.set_index('Timestamp (UTC)', inplace=True)
        weather_df.index = pd.to_datetime(weather_df.index, errors='coerce', utc=True)
        weather_df = weather_df[weather_df.index.notna()]
        weather_df = weather_df[weather_df.index.year == year]

        # Fail-fast merge: assume weather_df has the expected columns.
        # If not, let pandas raise immediately instead of silently falling back.
        df['GHI'] = weather_df['GHI (W/m^2)'].reindex(df.index)
        df['DHI'] = weather_df['DHI (W/m^2)'].reindex(df.index)
        df['DNI'] = weather_df['DNI (W/m^2)'].reindex(df.index)
        df['Solar Elevation Angle (Deg)'] = weather_df['Solar Elevation Angle (Deg)'].reindex(df.index)
        df['Solar Azimuth Angle (Deg)'] = weather_df['Solar Azimuth Angle (Deg)'].reindex(df.index)
        if 'Surface Albedo' in weather_df.columns:
            df['Surface Albedo'] = weather_df['Surface Albedo'].reindex(df.index)
        if 'apparent_zenith' in weather_df.columns:
            df['apparent_zenith'] = weather_df['apparent_zenith'].reindex(df.index)

        active_stow = df['trigger'].fillna('Ideal Tracking') != 'Ideal Tracking'
        filtered_df_as = df.loc[active_stow, ['avg_pos_angle', 'stow_angle']].dropna()
        mae_value_as = np.mean(np.abs(filtered_df_as['avg_pos_angle'] - filtered_df_as['stow_angle']))
        percent_as = (len(filtered_df_as) / len(df)) * 100 if len(df) else np.nan

        solar_position = loc.get_solarposition(df.index)
       
        dni_extra = pv.irradiance.get_extra_radiation(df.index)
        albedo = df['Surface Albedo'] if 'Surface Albedo' in df.columns else 0.2

        stow_surface = pv.tracking.calc_surface_orientation(df['stow_angle'], axis_tilt, axis_azimuth)
        avg_surface = pv.tracking.calc_surface_orientation(df['avg_pos_angle'], axis_tilt, axis_azimuth)

        stow_surface.to_csv(f'Final Version/{loc.name.lower().replace(" ", "_")}_stow_surface.csv')
        avg_surface.to_csv(f'Final Version/{loc.name.lower().replace(" ", "_")}_avg_surface.csv')
        solar_position.to_csv(f'Final Version/{loc.name.lower().replace(" ", "_")}_solar_position.csv')

        # stow_aoi = pv.irradiance.aoi(
        #     surface_tilt=stow_surface['surface_tilt'],
        #     surface_azimuth=stow_surface['surface_azimuth'],
        #     solar_zenith=solar_position['apparent_zenith'],
        #     solar_azimuth=solar_position['azimuth']
        # )
        # avg_aoi = pv.irradiance.aoi(
        #     surface_tilt=avg_surface['surface_tilt'],
        #     surface_azimuth=avg_surface['surface_azimuth'],
        #     solar_zenith=solar_position['apparent_zenith'],
        #     solar_azimuth=solar_position['azimuth']
        # )
        df.to_csv(f'Final Version/{loc.name.lower().replace(" ", "_")}_combined_eval.csv')
        poa_stow = pv.irradiance.get_total_irradiance(
            surface_tilt=stow_surface['surface_tilt'],
            surface_azimuth=stow_surface['surface_azimuth'],
            solar_zenith=solar_position['apparent_zenith'],
            solar_azimuth=solar_position['azimuth'],
            dni=df['DNI'],
            ghi=df['GHI'],
            dhi=df['DHI'],
            dni_extra=dni_extra,
            albedo=albedo
        )['poa_global']

        poa_avg_pos = pv.irradiance.get_total_irradiance(
            surface_tilt=avg_surface['surface_tilt'],
            surface_azimuth=avg_surface['surface_azimuth'],
            solar_zenith=solar_position['apparent_zenith'],
            solar_azimuth=solar_position['azimuth'],
            dni=df['DNI'],
            ghi=df['GHI'],
            dhi=df['DHI'],
            dni_extra=dni_extra,
            albedo=albedo
        )['poa_global']

        poa_stow.to_csv(f'Final Version/{loc.name.lower().replace(" ", "_")}_poa_stow.csv')
        poa_avg_pos.to_csv(f'Final Version/{loc.name.lower().replace(" ", "_")}_poa_avg_pos.csv')
        # Compare only on overlapping valid samples to avoid bias from data gaps.
        poa_pair = pd.DataFrame({
            'poa_stow': poa_stow,
            'poa_avg_pos': poa_avg_pos,
        }).dropna()

        poa_stow_total_year = poa_pair['poa_stow'].sum(min_count=1)
        poa_avg_pos_total_year = poa_pair['poa_avg_pos'].sum(min_count=1)
        poa_difference_total_year = poa_stow_total_year - poa_avg_pos_total_year

        stow_valid_samples = int(poa_stow.notna().sum())
        avg_valid_samples = int(poa_avg_pos.notna().sum())
        overlap_valid_samples = int(len(poa_pair))
        overlap_days = int(poa_pair.index.normalize().nunique()) if overlap_valid_samples else 0

        stow_days = int(poa_stow.dropna().index.normalize().nunique())
        avg_days = int(poa_avg_pos.dropna().index.normalize().nunique())

        return {
            'mae_active_stow': mae_value_as,
            'percent_active_stow': percent_as,
            'poa_stow_total_year': poa_stow_total_year,
            'poa_avg_pos_total_year': poa_avg_pos_total_year,
            'poa_difference_total_year': poa_difference_total_year,
            # 'poa_difference_abs_total_year': abs(poa_difference_total_year),
            'poa_%_difference_total_year': (poa_difference_total_year / poa_avg_pos_total_year) * 100,
            # 'poa_stow_valid_samples': stow_valid_samples,
            # 'poa_avg_pos_valid_samples': avg_valid_samples,
            # 'poa_overlap_valid_samples': overlap_valid_samples,
            # 'poa_stow_days_with_data': stow_days,
            # 'poa_avg_pos_days_with_data': avg_days,
            # 'poa_overlap_days_with_data': overlap_days,
        }

if __name__ == "__main__":
    # Toggle between the initial/example dataset (synthetic, no real tracker
    # ground truth) and the real-world validation sites (Site A-E).
    RUN_MODE = 'initial'  # 'initial' or 'validation'

    # See DEFAULT_STOW_TOGGLES for the full set of keys and their meaning.
    ST_No_Zero_No_Preemptive = {
        # No zero-crossing and no preemptive stow toggles for the ST_No_Zero_No_Preemptive configuration.
        **DEFAULT_STOW_TOGGLES
    }

    ST_Zero_Preemptive = {
        # Zero-crossing and preemptive stow toggles for the ST_Zero_Preemptive configuration.
        'enable_storm': True,
        'enable_wind': True,
        'enable_snow': True,
        'storm_zero_crossing': True,
        'wind_zero_crossing': True,
        'snow_zero_crossing': False,
        'preempt_storm': True,
        'preempt_snow': True,
        'preempt_steps': 2,
    }

    ST_Zero_No_Preemptive = {
        # Zero-crossing and no preemptive stow toggles
        'enable_storm': True,
        'enable_wind': True,
        'enable_snow': True,
        'storm_zero_crossing': True,
        'wind_zero_crossing': True,
        'snow_zero_crossing': False,
        'preempt_storm': False,
        'preempt_snow': False,
        'preempt_steps': 2,
    }

    ST_No_Zero_Preemptive = {
        # No zero-crossing and preemptive stow toggles
        'enable_storm': True,
        'enable_wind': True,
        'enable_snow': True,
        'storm_zero_crossing': False,
        'wind_zero_crossing': False,
        'snow_zero_crossing': False,
        'preempt_storm': True,
        'preempt_snow': True,
        'preempt_steps': 2,
    }

    if RUN_MODE == 'initial':
        # Give each toggle combination you want to compare its own output_name
        # (e.g. 'site_example_baseline', 'site_example_preemptive') so the CSVs
        # don't overwrite each other.
        Site.run_initial_site(stow_toggles=ST_No_Zero_No_Preemptive, output_name='site_example_no_zero_no_preemptive_stow')
        Site.run_initial_site(stow_toggles=ST_Zero_Preemptive, output_name='site_example_zero_preemptive')
        Site.run_initial_site(stow_toggles=ST_Zero_No_Preemptive, output_name='site_example_zero_no_preemptive')
        Site.run_initial_site(stow_toggles=ST_No_Zero_Preemptive, output_name='site_example_no_zero_preemptive')

    elif RUN_MODE == 'validation':
        x = False
        if x:
            Site.runall(stow_toggles=STOW_TOGGLES)
        else:
            pass

        sites = [
            'Site A', 'Site B', 'Site C', 'Site D', 'Site E']
        site_metrics = {}

        for site in sites:
            if site == 'Site A':
                year = 2023
            else:
                year = 2024

            location = pv.location.Location(
                latitude=SITE_CONFIGS[site.lower().replace(" ", "_")]['latitude'],
                longitude=SITE_CONFIGS[site.lower().replace(" ", "_")]['longitude'],
                tz=SITE_CONFIGS[site.lower().replace(" ", "_")]['tz'],
                altitude=SITE_CONFIGS[site.lower().replace(" ", "_")]['altitude'],
                name=site
            )
            
            raw_metrics = Site.stow_metrics(
                fp=f'Final Version/Output/{site.lower().replace(" ", "_")}_combined.csv',
                weather_fp=f'Final Version/WeatherBit Data/{site}.csv',
                year = year,
                loc = location
            )

            rounded_metrics = {
                key: (round(float(value), 3) if pd.notna(value) else np.nan)
                for key, value in raw_metrics.items()
            }
            site_metrics[site] = rounded_metrics

            print(site)
            print(rounded_metrics)

        metrics_table = pd.DataFrame.from_dict(site_metrics, orient='index')
        metrics_table.index.name = 'site'
        print('\nStow Metrics Summary (rounded to 3 decimals):')
        print(metrics_table.to_string(float_format=lambda x: f'{x:.3f}', na_rep='NaN'))

    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE!r}. Expected 'initial' or 'validation'.")