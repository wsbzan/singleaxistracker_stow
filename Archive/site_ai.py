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
    
    def run_stow_conditions(self, df, relaxation_factor, max_angle):
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

        for i in range(len(df) - 1):
            next_i = i + 1
            has_trigger = not pd.isna(trigger[i])
            wc = weather_code[i]

            if wind_speed[i] > 25 or wc in storm_codes:
                if not has_trigger:
                    trigger[i] = 'Storm'
                    relaxation_factor = 40
                    stow_setpoint[next_i] = -max_angle if stow_angle[i] < 0 else max_angle
            elif wind_speed[i] > 11 or wind_gust[i] > 16:
                if not has_trigger:
                    trigger[i] = 'Wind'
                    relaxation_factor = 20
                    stow_setpoint[next_i] = -40 if stow_angle[i] < 0 else 40
            elif snowfall[i] > 0 or wc in snow_codes:
                if not has_trigger:
                    trigger[i] = 'Snow'
                    relaxation_factor = 30
                    stow_setpoint[next_i] = -max_angle if wind_dir[i] <= 180 else max_angle
            elif relaxation_factor > 0:
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
        site_tz = getattr(self, 'tz')
        # stow timestamps are UTC, convert to local site time
        stow_df.index = (
            pd.to_datetime(stow_df.index, errors="coerce")
            .tz_localize("UTC")
            .tz_convert(site_tz)
            .tz_localize(None)
        )

        tracker_idx = pd.to_datetime(tracker_df.index, errors='coerce')
        if getattr(tracker_idx, 'tz', None) is not None:
            tracker_idx = tracker_idx.tz_convert(site_tz).tz_localize(None)
        tracker_df.index = tracker_idx

        stow_df.index = pd.to_datetime(stow_df.index, errors='coerce')

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
    def runall():
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

            stow_conditions = site_obj.process_stow_conditions()
            stow_out = os.path.join(output_dir, f'{site_key}_stow_conditions.csv')
            stow_conditions.to_csv(stow_out)

            combined = site_obj.combine_tracker_positions(stow_out, nexamp_path)
            combined_out = os.path.join(output_dir, f'{site_key}_combined.csv')
            combined.to_csv(combined_out)

            scatter_out = os.path.join(output_dir, f'{site_key}_scatter.png')
            site_obj.plot_combined_scatter(combined, scatter_out)

            print(f'Finished {site_key}: {combined_out}, {scatter_out}')

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
    def stow_metrics(fp, weather_fp=None, axis_tilt=0, axis_azimuth=180):
        df = pd.read_csv(fp, parse_dates=['timestamp']).set_index('timestamp')
        df['avg_pos_angle'] = df[['pos1', 'pos2', 'pos3']].mean(axis=1)

        def _first_existing(columns, candidates, required=True):
            for candidate in candidates:
                if candidate in columns:
                    return candidate
            if required:
                raise ValueError(f"Missing required column. Expected one of: {', '.join(candidates)}")
            return None

        if weather_fp is not None:
            weather_df = pd.read_csv(weather_fp)
            if 'Timestamp (Local)' in weather_df.columns:
                weather_df['timestamp'] = pd.to_datetime(weather_df['Timestamp (Local)'], errors='coerce')
            elif 'timestamp' in weather_df.columns:
                weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], errors='coerce')
            elif 'Timestamp (UTC)' in weather_df.columns:
                weather_df['timestamp'] = pd.to_datetime(weather_df['Timestamp (UTC)'], errors='coerce')
            else:
                raise ValueError('weather_fp must include a timestamp column such as Timestamp (Local) or Timestamp (UTC).')

            weather_df.set_index('timestamp', inplace=True)
            weather_df.index = pd.to_datetime(weather_df.index, errors='coerce')
            weather_df = weather_df[weather_df.index.notna()]

            weather_column_map = {
                'GHI': ['GHI', 'GHI (W/m^2)'],
                'DHI': ['DHI', 'DHI (W/m^2)'],
                'DNI': ['DNI', 'DNI (W/m^2)'],
                'Solar Elevation Angle (Deg)': ['Solar Elevation Angle (Deg)'],
                'Solar Azimuth Angle (Deg)': ['Solar Azimuth Angle (Deg)'],
                'Surface Albedo': ['Surface Albedo'],
                'apparent_zenith': ['apparent_zenith']
            }

            for target_column, candidate_columns in weather_column_map.items():
                if target_column not in df.columns:
                    source_column = _first_existing(weather_df.columns, candidate_columns, required=False)
                    if source_column is not None:
                        df[target_column] = weather_df[source_column].reindex(df.index)

        active_stow = df['trigger'].fillna('Ideal Tracking') != 'Ideal Tracking'
        filtered_df_as = df.loc[active_stow, ['avg_pos_angle', 'stow_angle']].dropna()
        mae_value_as = np.mean(np.abs(filtered_df_as['avg_pos_angle'] - filtered_df_as['stow_angle']))
        percent_as = (len(filtered_df_as) / len(df)) * 100 if len(df) else np.nan

        required_weather_columns = [
            'GHI',
            'DHI',
            'DNI',
            'Solar Azimuth Angle (Deg)'
        ]
        missing_weather = [column for column in required_weather_columns if column not in df.columns]
        if missing_weather:
            raise ValueError(
                'Missing irradiance inputs for the stow comparison: '
                + ', '.join(missing_weather)
                + '. Provide weather_fp or pre-merge these columns into the input file.'
            )

        solar_azimuth = df['Solar Azimuth Angle (Deg)']
        if 'apparent_zenith' in df.columns:
            solar_zenith = df['apparent_zenith']
        elif 'Solar Elevation Angle (Deg)' in df.columns:
            solar_zenith = 90 - df['Solar Elevation Angle (Deg)']
        else:
            raise ValueError(
                'Missing solar position inputs for the stow comparison. '
                'Provide either apparent_zenith or Solar Elevation Angle (Deg).'
            )

        dni_extra = pv.irradiance.get_extra_radiation(df.index)
        albedo = df['Surface Albedo'] if 'Surface Albedo' in df.columns else 0.2

        stow_surface = pv.tracking.calc_surface_orientation(df['stow_angle'], axis_tilt, axis_azimuth)
        avg_surface = pv.tracking.calc_surface_orientation(df['avg_pos_angle'], axis_tilt, axis_azimuth)

        poa_stow = pv.irradiance.get_total_irradiance(
            surface_tilt=stow_surface['surface_tilt'],
            surface_azimuth=stow_surface['surface_azimuth'],
            solar_zenith=solar_zenith,
            solar_azimuth=solar_azimuth,
            dni=df['DNI'],
            ghi=df['GHI'],
            dhi=df['DHI'],
            dni_extra=dni_extra,
            albedo=albedo
        )['poa_global']

        poa_avg_pos = pv.irradiance.get_total_irradiance(
            surface_tilt=avg_surface['surface_tilt'],
            surface_azimuth=avg_surface['surface_azimuth'],
            solar_zenith=solar_zenith,
            solar_azimuth=solar_azimuth,
            dni=df['DNI'],
            ghi=df['GHI'],
            dhi=df['DHI'],
            dni_extra=dni_extra,
            albedo=albedo
        )['poa_global']

        active_poa_delta = (poa_stow - poa_avg_pos).loc[active_stow].dropna()

        return {
            'mae_active_stow': mae_value_as,
            'percent_active_stow': percent_as,
            'poa_stow_mean_active_stow': poa_stow.loc[active_stow].mean(),
            'poa_avg_pos_mean_active_stow': poa_avg_pos.loc[active_stow].mean(),
            'poa_difference_mean_active_stow': active_poa_delta.mean(),
            'poa_difference_abs_mean_active_stow': active_poa_delta.abs().mean()
        }

if __name__ == "__main__":
    Site.runall()
    print('Site A')
    print(Site.stow_metrics(fp='Final Version/Output/site_a_combined.csv',
                            weather_fp='Final Version/WeatherBit Data/Site A.csv'))
    # print('Site B')
    # print(Site.mae(fp='Final Version/Output/site_b_combined.csv'))
    # print('Site C')
    # print(Site.mae(fp='Final Version/Output/site_c_combined.csv'))