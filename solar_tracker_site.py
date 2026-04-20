import pandas as pd
import numpy as np
import pvlib
import matplotlib.pyplot as plt
from site_info import site_info as si
from api_calls import api_call
import os

class SolarTrackerSite:
    """
    A comprehensive class for modeling single-axis solar tracker systems with stow conditions.
    Handles weather data import, PV system modeling, stow logic, and results analysis.
    """

    def __init__(self, site_config=None):
        """
        Initialize the SolarTrackerSite with configuration parameters.

        Args:
            site_config (dict): Site configuration dictionary. If None, uses site_info from site_info.py
        """
        self.site_config = site_config if site_config else si
        self.location = None
        self.mount = None
        self.array = None
        self.system = None
        self.modelchain = None
        self.input_df = None
        self.solar_position = None
        self.tracker_angles_ideal = None
        self.tracker_angles_stow = None
        self.results = None
        self.weather_data = {}

    def setup_location(self):
        """Establish PVlib Location Object"""
        self.location = pvlib.location.Location(
            latitude=self.site_config['latitude'],
            longitude=self.site_config['longitude'],
            tz=self.site_config['tz'],
            altitude=self.site_config['altitude'],
            name=self.site_config['name']
        )

    def import_weather_data(self, psm_file=None, weatherbit_file=None, use_api=False):
        """
        Import weather data from files or API calls.

        Args:
            psm_file (str): Path to PSM4 CSV file
            weatherbit_file (str): Path to Weatherbit CSV file
            use_api (bool): Whether to fetch data from APIs
        """
        # Main weather data (PSM4)
        if use_api:
            api = api_call()
            psm4_data = api.fetch_psm4_data()
            with open('psm4_data.csv', 'w') as file:
                file.write(psm4_data)
            self.input_df = pd.read_csv('psm4_data.csv', index_col=0, parse_dates=True)
        else:
            psm_file = psm_file or self.site_config['psm_file']
            self.input_df = pd.read_csv(psm_file, parse_dates=True, skiprows=2)
            self.input_df['timestamp_local'] = pd.to_datetime(
                self.input_df[['Year','Month','Day','Hour','Minute']]
            )
            self.input_df.set_index('timestamp_local', inplace=True)

        # Weatherbit data for stow conditions
        if use_api:
            api = api_call()
            weatherbit_data = api.fetch_weatherbit_data()
            df = pd.json_normalize(weatherbit_data, 'data',
                                 ["city_id","city_name","country_code","lat","lon",
                                  "state_code","station_id","timezone"])
            df.set_index(pd.to_datetime(df['timestamp_local']), inplace=True)
            df.rename(columns={
                'temp':'temp_air',
                'Wind Speed (m/s)':'wind_speed',
                'precipitable_water':'precipitable_water'
            }, inplace=True)
            self.weather_df = df[['temp_air','wind_speed','precipitable_water']]
            self.weather_df.to_csv('weatherbit_data.csv')
        else:
            weatherbit_file = weatherbit_file or self.site_config['weatherbit_file']
            self.weather_df = pd.read_csv(weatherbit_file, parse_dates=True)
            self.weather_df.set_index(pd.to_datetime(self.weather_df['Timestamp (Local)']), inplace=True)

        # Merge weather data
        self.input_df = pd.concat([self.input_df, self.weather_df], axis=1)

    def setup_pv_system(self):
        """Build PV system components (mount, array, system, modelchain)"""
        # Temperature model
        temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']

        # PV module and inverter models
        cec_module_db = pvlib.pvsystem.retrieve_sam('cecmod')
        module_parameters = cec_module_db[self.site_config['module_name']]
        module_parameters['Technology'] = 'CdTe'  # Spectral correction

        cec_inverter_db = pvlib.pvsystem.retrieve_sam('cecinverter')
        inverter_parameters = cec_inverter_db[self.site_config['inverter__name']]

        # Build components
        self.mount = pvlib.pvsystem.SingleAxisTrackerMount(
            axis_tilt=self.site_config['axis_tilt'],
            axis_azimuth=self.site_config['axis_azimuth'],
            max_angle=self.site_config['max_angle'],
            backtrack=self.site_config['backtrack']
        )

        self.array = pvlib.pvsystem.Array(
            mount=self.mount,
            module_parameters=module_parameters,
            temperature_model_parameters=self.site_config['temperature_model_parameters'],
            modules_per_string=self.site_config['modules_per_string'],
            strings=self.site_config['strings_per_inverter']
        )

        self.system = pvlib.pvsystem.PVSystem(
            arrays=[self.array], inverter_parameters=inverter_parameters
        )

        self.modelchain = pvlib.modelchain.ModelChain(
            self.system,
            self.location,
            ac_model='sandia',
            aoi_model='physical'
        )

    def calculate_solar_position(self):
        """Calculate solar position for the input timestamps"""
        if self.input_df is None:
            raise ValueError("Weather data must be imported first")
        self.solar_position = self.location.get_solarposition(self.input_df.index)

    def get_ideal_tracker_angles(self):
        """Calculate ideal tracker angles without stow conditions"""
        if self.mount is None or self.solar_position is None:
            raise ValueError("PV system and solar position must be calculated first")

        self.tracker_angles_ideal = self.mount.get_orientation(
            self.solar_position['apparent_zenith'],
            self.solar_position['azimuth']
        )
        self.tracker_angles_ideal['tracker_theta'] = self.tracker_angles_ideal['tracker_theta'].ffill()

        return self.tracker_angles_ideal

    def run_stow_conditions(self, tracker_df):
        """
        Apply stow conditions logic to tracker angles.

        Args:
            tracker_df (pd.DataFrame): DataFrame with tracker angles and weather data

        Returns:
            pd.DataFrame: DataFrame with stow angles and triggers
        """
        # Initialize stow columns
        tracker_df.insert(0,'trigger',np.nan)
        tracker_df['trigger'] = tracker_df['trigger'].astype('object')
        tracker_df.insert(1,'stow_angle',np.nan)
        tracker_df.insert(2,'stow_setpoint',np.nan)

        # Time calculations
        tracker_df['td'] = pd.to_datetime(tracker_df.index)
        tracker_df['time_delta'] = (tracker_df['td']-tracker_df['td'].shift())

        relaxation_factor = 0
        max_angle = self.site_config['max_angle']

        # Set initial conditions
        tracker_df.at[tracker_df.index[0], 'stow_setpoint'] = tracker_df.at[tracker_df.index[0], 'tracker_theta']
        tracker_df.at[tracker_df.index[0], 'stow_angle'] = tracker_df.at[tracker_df.index[0], 'tracker_theta']

        for i in range(len(tracker_df)-1):
            idx = tracker_df.index[i]
            next_idx = tracker_df.index[i+1]
            row = tracker_df.loc[idx]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            # Reset trigger for this timestep
            trigger_set = False

            # Storm conditions
            if (row['Wind Speed (m/s)'] > 25 or
                row['Weather Code'] in [200,201,202,230,231,232,233,511]):
                if pd.isna(tracker_df.at[idx, 'trigger']):
                    tracker_df.at[idx, 'trigger'] = "Storm"
                    relaxation_factor = 40
                    trigger_set = True
                    if row['stow_angle'] < 0:
                        tracker_df.at[next_idx, 'stow_setpoint'] = -max_angle
                    else:
                        tracker_df.at[next_idx, 'stow_setpoint'] = max_angle

            # Wind conditions
            elif (row['Wind Speed (m/s)'] > 11 or row['Wind Gust Speed (m/s)'] > 16):
                if pd.isna(tracker_df.at[idx, 'trigger']):
                    tracker_df.at[idx, 'trigger'] = "Wind"
                    relaxation_factor = 20
                    trigger_set = True
                    if row['stow_angle'] < 0:
                        tracker_df.at[next_idx, 'stow_setpoint'] = -40
                    else:
                        tracker_df.at[next_idx, 'stow_setpoint'] = 40

            # Snow conditions
            elif (row['Snowfall Rate (mm/hr)'] > 0 or
                  row['Weather Code'] in [600,601,602,610,611,612,621,622,623]):
                if pd.isna(tracker_df.at[idx, 'trigger']):
                    tracker_df.at[idx, 'trigger'] = "Snow"
                    relaxation_factor = 30
                    trigger_set = True
                    # Wind direction determines stow direction
                    if row['Wind Dir (Deg)'] <= 180:
                        tracker_df.at[next_idx, 'stow_setpoint'] = -max_angle
                    else:
                        tracker_df.at[next_idx, 'stow_setpoint'] = max_angle

            # Relaxation period
            elif relaxation_factor > 0:
                tracker_df.at[idx, 'trigger'] = "Relaxing"
                time_delta = tracker_df.at[idx, 'time_delta'].total_seconds() / 60
                relaxation_factor -= time_delta
                if relaxation_factor > 0:
                    tracker_df.at[next_idx, 'stow_setpoint'] = tracker_df.at[idx, 'stow_setpoint']
                else:
                    tracker_df.at[next_idx, 'stow_setpoint'] = tracker_df.at[next_idx, 'tracker_theta']

            # Normal tracking
            else:
                tracker_df.at[next_idx, 'stow_setpoint'] = tracker_df.at[next_idx, 'tracker_theta']
                tracker_df.at[idx, 'trigger'] = "Ideal Tracking"

            # Calculate angle movement constraints
            if not pd.isna(tracker_df.at[next_idx, 'stow_setpoint']):
                setpoint_angle = tracker_df.at[next_idx, 'stow_setpoint']
                current_angle = tracker_df.at[idx, 'stow_angle']

                angle_delta = setpoint_angle - current_angle
                max_angle_change = 30  # degrees per 15-minute timestep

                if abs(angle_delta) > max_angle_change:
                    if setpoint_angle < current_angle:
                        tracker_df.at[next_idx, 'stow_angle'] = max(current_angle - max_angle_change, -max_angle)
                    else:
                        tracker_df.at[next_idx, 'stow_angle'] = min(current_angle + max_angle_change, max_angle)
                else:
                    tracker_df.at[next_idx, 'stow_angle'] = setpoint_angle

        return tracker_df

    def apply_stow_conditions(self):
        """Apply stow conditions to ideal tracker angles"""
        if self.tracker_angles_ideal is None:
            raise ValueError("Ideal tracker angles must be calculated first")

        # Combine tracker angles with weather data
        tracker_df = pd.concat([self.tracker_angles_ideal, self.input_df], axis=1)

        # Apply stow logic
        self.tracker_angles_stow = self.run_stow_conditions(tracker_df)

        return self.tracker_angles_stow

    def recalculate_aoi_and_poa(self, tracker_angles_df):
        """
        Recalculate AOI and POA for stow angles.

        Args:
            tracker_angles_df (pd.DataFrame): DataFrame with stow angles

        Returns:
            pd.DataFrame: Updated DataFrame with recalculated AOI and surface orientation
        """
        sol_pos = self.solar_position.copy()
        sol_pos.index = sol_pos.index.tz_localize(None)

        surface = pvlib.tracking.calc_surface_orientation(
            tracker_angles_df['stow_angle'],
            self.site_config['axis_tilt'],
            self.site_config['axis_azimuth']
        )

        aoi = pvlib.irradiance.aoi(
            surface['surface_tilt'],
            surface['surface_azimuth'],
            sol_pos['apparent_zenith'],
            sol_pos['azimuth']
        )

        aoi = aoi[~aoi.index.duplicated(keep='first')]
        tracker_angles_df['aoi'] = aoi
        tracker_angles_df['surface_tilt'] = surface['surface_tilt']
        tracker_angles_df['surface_azimuth'] = surface['surface_azimuth']

        return tracker_angles_df

    def build_weather_data_for_power(self, tracker_angles_df, scenario_name=""):
        """
        Build weather data for power calculation using specific tracker angles.

        Args:
            tracker_angles_df (pd.DataFrame): Tracker angles DataFrame
            scenario_name (str): Name for this scenario (for saving results)

        Returns:
            pd.DataFrame: Weather data for modelchain
        """
        dni_extra = pvlib.irradiance.get_extra_radiation(self.solar_position.index)

        averaged_irradiance = pvlib.bifacial.infinite_sheds.get_irradiance_poa(
            tracker_angles_df['surface_tilt'].values,
            tracker_angles_df['surface_azimuth'].values,
            self.solar_position['apparent_zenith'].values,
            self.solar_position['azimuth'].values,
            self.site_config['gcr'],
            self.site_config['axis_height'],
            self.site_config['pitch'],
            self.input_df['GHI'].values,
            self.input_df['DHI'].values,
            self.input_df['DNI'].values,
            self.input_df['Surface Albedo'].values,
            model='haysdavies',
            dni_extra=dni_extra
        )

        cell_temperature_steady_state = pvlib.temperature.faiman(
            poa_global=averaged_irradiance['poa_global'],
            temp_air=self.input_df['Temperature'],
            wind_speed=self.input_df['Wind Speed'],
            **self.site_config['temperature_model_parameters'],
        )

        cell_temperature_steady_state.index = pd.to_datetime(
            cell_temperature_steady_state.index, format="mixed", errors='coerce'
        )
        cell_temperature_steady_state.fillna(0, inplace=True)

        cell_temperature = pvlib.temperature.prilliman(
            cell_temperature_steady_state,
            self.input_df['Wind Speed'],
            unit_mass=self.site_config['module_unit_mass']
        )

        weather_data = pd.DataFrame({
            'poa_global': averaged_irradiance['poa_global'],
            'poa_direct': averaged_irradiance['poa_direct'],
            'poa_diffuse': averaged_irradiance['poa_diffuse'],
            'cell_temperature': cell_temperature,
            'precipitable_water': self.input_df['Precipitable Water'].values
        })

        self.weather_data[scenario_name] = weather_data
        return weather_data

    def run_power_model(self, weather_data):
        """
        Run the PV modelchain to calculate power output.

        Args:
            weather_data (pd.DataFrame): Weather data for the model

        Returns:
            dict: AC and DC power results
        """
        self.modelchain.run_model_from_poa(weather_data)

        ac_power = self.modelchain.results.ac / 1000  # Convert to kW
        dc_power = self.modelchain.results.dc['p_mp'] / 1000  # Convert to kW

        return {
            'ac': ac_power,
            'dc': dc_power
        }

    def run_complete_analysis(self, save_intermediate=True):
        """
        Run the complete analysis pipeline.

        Args:
            save_intermediate (bool): Whether to save intermediate results
        """
        print("Setting up location...")
        self.setup_location()

        print("Importing weather data...")
        self.import_weather_data()

        print("Setting up PV system...")
        self.setup_pv_system()

        print("Calculating solar position...")
        self.calculate_solar_position()

        print("Calculating ideal tracker angles...")
        self.get_ideal_tracker_angles()

        print("Applying stow conditions...")
        self.apply_stow_conditions()

        if save_intermediate:
            self.save_tracker_angles()

        print("Recalculating AOI and POA...")
        self.tracker_angles_stow = self.recalculate_aoi_and_poa(self.tracker_angles_stow)

        print("Running power analysis...")
        self.results = {}

        # Ideal tracking scenario
        print("  - Ideal tracking...")
        weather_ideal = self.build_weather_data_for_power(self.tracker_angles_ideal, "ideal")
        power_ideal = self.run_power_model(weather_ideal)
        self.results['ideal'] = power_ideal

        # Stow scenario
        print("  - With stow conditions...")
        weather_stow = self.build_weather_data_for_power(self.tracker_angles_stow, "stow")
        power_stow = self.run_power_model(weather_stow)
        self.results['stow'] = power_stow

        print("Analysis complete!")
        self.print_summary()

    def print_summary(self):
        """Print summary of results"""
        if self.results:
            print("\n" + "="*50)
            print("RESULTS SUMMARY")
            print("="*50)

            for scenario, power in self.results.items():
                total_dc = power['dc'].sum()
                total_ac = power['ac'].sum()
                print(".2f"
                      ".2f")

            # Energy loss due to stow
            dc_loss = self.results['ideal']['dc'].sum() - self.results['stow']['dc'].sum()
            ac_loss = self.results['ideal']['ac'].sum() - self.results['stow']['ac'].sum()
            print(".2f"
                  ".2f")

    def save_results(self, output_dir="results"):
        """
        Save all results for charting and analysis.

        Args:
            output_dir (str): Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.results:
            # Save power output data
            for scenario, power in self.results.items():
                power['dc'].to_csv(f'{output_dir}/{scenario}_dc_output.csv')
                power['ac'].to_csv(f'{output_dir}/{scenario}_ac_output.csv')

        # Save tracker angles
        if self.tracker_angles_ideal is not None:
            self.tracker_angles_ideal.to_csv(f'{output_dir}/tracker_angles_ideal.csv')

        if self.tracker_angles_stow is not None:
            self.tracker_angles_stow.to_csv(f'{output_dir}/tracker_angles_stow.csv')

        # Save weather data
        for scenario, weather in self.weather_data.items():
            weather.to_csv(f'{output_dir}/weather_data_{scenario}.csv')

        print(f"Results saved to {output_dir}/")

    def save_tracker_angles(self):
        """Save tracker angles to CSV files"""
        if self.tracker_angles_stow is not None:
            self.tracker_angles_stow.to_csv('tracker_angles_with_stow_conditions.csv')

    def filter_by_date_range(self, start_date, end_date, date_col='timestamp_local'):
        """
        Filter tracker angles DataFrame by date range.

        Args:
            start_date (str): Start date in format 'MM/DD/YY'
            end_date (str): End date in format 'MM/DD/YY'
            date_col (str): Column name containing dates

        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        if self.tracker_angles_stow is None:
            raise ValueError("Stow conditions must be applied first")

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self.tracker_angles_stow[date_col]):
            self.tracker_angles_stow[date_col] = pd.to_datetime(self.tracker_angles_stow[date_col])

        # Filter by date range
        filtered_df = self.tracker_angles_stow[
            (self.tracker_angles_stow[date_col].dt.strftime('%m/%d/%y') >= start_date) &
            (self.tracker_angles_stow[date_col].dt.strftime('%m/%d/%y') <= end_date)
        ]

        return filtered_df

    def create_stow_timeline_plot(self, save_path='stow_conditions_timeline.png'):
        """
        Create a timeline plot showing stow condition triggers.

        Args:
            save_path (str): Path to save the plot
        """
        if self.tracker_angles_stow is None:
            raise ValueError("Stow conditions must be applied first")

        # Fill NaN triggers
        plot_df = self.tracker_angles_stow.copy()
        plot_df['trigger'].fillna("Ideal Tracking", inplace=True)
        plot_df = plot_df[~plot_df['trigger'].isna()]

        # Color mapping
        manual_color_map = {
            'Ideal Tracking': 'gray',
            'Storm': 'red',
            'Wind': 'blue',
            'Snow': 'cyan',
            'Relaxing': 'orange'
        }

        fig, ax = plt.subplots(figsize=(12, 2))

        # Plot each timestamp as a colored bar
        for i, (timestamp, row) in enumerate(plot_df.iterrows()):
            trigger = row['trigger']
            color = manual_color_map.get(trigger, 'black')
            ax.bar(timestamp, 1, color=color, width=0.01)

        # Formatting
        ax.set_xlim(plot_df.index.min(), plot_df.index.max())
        ax.set_xlabel('Time')
        ax.set_yticks([])

        # Legend
        handles = [plt.Line2D([0], [0], color=color, lw=4)
                  for color in manual_color_map.values()]
        ax.legend(handles, list(manual_color_map.keys()),
                 title='Trigger', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Timeline plot saved to {save_path}")


if __name__ == '__main__':
    # Example usage
    site = SolarTrackerSite()

    # Run complete analysis
    site.run_complete_analysis()

    # Save results for charting
    site.save_results()

    # Create timeline plot
    site.create_stow_timeline_plot()

    # Filter for specific date range (example: April 2-3, 2024)
    filtered_data = site.filter_by_date_range('04/02/24', '04/03/24')
    filtered_data.to_csv('april_2_and_3_example.csv')
    print(f"Filtered data saved to april_2_and_3_example.csv ({len(filtered_data)} rows)")