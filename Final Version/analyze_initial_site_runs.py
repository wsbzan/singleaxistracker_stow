"""
Analyze the stow_conditions CSVs produced by Site.run_initial_site() in
site_ai_v2.py (the synthetic 'site_example' dataset, which has no real
tracker ground truth to compare against).

For each CSV this reports:
  1) Number of total/active/preemptive stow triggers, counted as every
     timestep the trigger fires (repeats of the same trigger count too).
  2) Time spent in active stow vs preemptive stow vs relaxing vs ideal tracking.
  3) Irradiance captured (POA energy) using the stow angle vs the ideal
     tracker angle (tracker_theta), since there is no real tracker to use
     instead (unlike Site.stow_metrics, which compares against avg_pos_angle).
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import pvlib as pv

from sites import sites as SITE_CONFIGS

ACTIVE_TRIGGERS = ['Storm', 'Wind', 'Snow']
PREEMPTIVE_TRIGGERS = ['Preemptive Storm', 'Preemptive Snow']
RELAXING_TRIGGER = 'Relaxing'
IDEAL_TRIGGER = 'Ideal Tracking'


def load_stow_conditions(fp):
    df = pd.read_csv(fp, parse_dates=['Timestamp (UTC)']).set_index('Timestamp (UTC)')
    df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
    df = df[df.index.notna()].sort_index()

    df['time_delta'] = pd.to_timedelta(df['time_delta'], errors='coerce')
    # Last row never gets a trigger assigned by run_stow_conditions; treat it as ideal tracking.
    df['trigger'] = df['trigger'].fillna(IDEAL_TRIGGER)

    if 'timestamp_local' in df.columns:
        df['timestamp_local'] = pd.to_datetime(df['timestamp_local'], errors='coerce')

    for col in ['stow_angle', 'tracker_theta', 'GHI (W/m^2)', 'DNI (W/m^2)',
                'DHI (W/m^2)', 'Surface Albedo']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def filter_date_range(df, start_date, end_date):
    """Restrict df to [start_date, end_date] inclusive, using local time when available."""
    if start_date is None and end_date is None:
        return df

    local_time = df['timestamp_local'] if 'timestamp_local' in df.columns else df.index.tz_localize(None)
    mask = pd.Series(True, index=df.index)
    if start_date is not None:
        mask &= local_time >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= local_time < pd.Timestamp(end_date) + pd.Timedelta(days=1)
    return df.loc[mask]


def trigger_counts(df):
    """Count every timestep a trigger fires, including repeats of the same trigger."""
    counts = df['trigger'].value_counts()

    active = int(counts.reindex(ACTIVE_TRIGGERS, fill_value=0).sum())
    preemptive = int(counts.reindex(PREEMPTIVE_TRIGGERS, fill_value=0).sum())

    return {
        'triggers_total': active + preemptive,
        'triggers_active': active,
        'triggers_preemptive': preemptive,
        'triggers_storm': int(counts.get('Storm', 0)),
        'triggers_wind': int(counts.get('Wind', 0)),
        'triggers_snow': int(counts.get('Snow', 0)),
        'triggers_preemptive_storm': int(counts.get('Preemptive Storm', 0)),
        'triggers_preemptive_snow': int(counts.get('Preemptive Snow', 0)),
    }


def time_breakdown(df):
    total_hours = df['time_delta'].dt.total_seconds().sum() / 3600
    hours_by_trigger = df.groupby('trigger')['time_delta'].apply(
        lambda s: s.dt.total_seconds().sum() / 3600)

    active_hours = hours_by_trigger.reindex(ACTIVE_TRIGGERS, fill_value=0).sum()
    preemptive_hours = hours_by_trigger.reindex(PREEMPTIVE_TRIGGERS, fill_value=0).sum()
    relaxing_hours = hours_by_trigger.get(RELAXING_TRIGGER, 0)
    stow_hours = active_hours + preemptive_hours + relaxing_hours
    ideal_hours = hours_by_trigger.get(IDEAL_TRIGGER, 0)

    return {
        'hours_total': total_hours,
        'hours_stow_or_relaxing': stow_hours,
        'hours_active_stow': active_hours,
        'hours_preemptive_stow': preemptive_hours,
        'hours_relaxing': relaxing_hours,
        'hours_ideal_tracking': ideal_hours,
        'pct_stow_or_relaxing': 100 * stow_hours / total_hours if total_hours else np.nan,
        'pct_active_stow': 100 * active_hours / total_hours if total_hours else np.nan,
        'pct_preemptive_stow': 100 * preemptive_hours / total_hours if total_hours else np.nan,
        'pct_ideal_tracking': 100 * ideal_hours / total_hours if total_hours else np.nan,
    }


def irradiance_capture(df, site_params):
    loc = pv.location.Location(
        latitude=site_params['latitude'],
        longitude=site_params['longitude'],
        tz=site_params['tz'],
        altitude=site_params['altitude'],
    )
    axis_tilt = site_params['axis_tilt']
    axis_azimuth = site_params['axis_azimuth']

    solar_position = loc.get_solarposition(df.index)
    dni_extra = pv.irradiance.get_extra_radiation(df.index)
    albedo = df['Surface Albedo'].fillna(0.2) if 'Surface Albedo' in df.columns else 0.2

    stow_surface = pv.tracking.calc_surface_orientation(df['stow_angle'], axis_tilt, axis_azimuth)
    ideal_surface = pv.tracking.calc_surface_orientation(df['tracker_theta'], axis_tilt, axis_azimuth)

    def poa_global(surface):
        return pv.irradiance.get_total_irradiance(
            surface_tilt=surface['surface_tilt'],
            surface_azimuth=surface['surface_azimuth'],
            solar_zenith=solar_position['apparent_zenith'],
            solar_azimuth=solar_position['azimuth'],
            dni=df['DNI (W/m^2)'],
            ghi=df['GHI (W/m^2)'],
            dhi=df['DHI (W/m^2)'],
            dni_extra=dni_extra,
            albedo=albedo,
        )['poa_global']

    hours = df['time_delta'].dt.total_seconds() / 3600
    pair = pd.DataFrame({
        'poa_stow': poa_global(stow_surface),
        'poa_ideal': poa_global(ideal_surface),
        'hours': hours,
    }).dropna()

    energy_stow = (pair['poa_stow'] * pair['hours']).sum()
    energy_ideal = (pair['poa_ideal'] * pair['hours']).sum()
    diff = energy_stow - energy_ideal

    return {
        'poa_energy_stow_Wh_m2': energy_stow,
        'poa_energy_ideal_Wh_m2': energy_ideal,
        'poa_energy_diff_Wh_m2': diff,
        'poa_energy_pct_diff': 100 * diff / energy_ideal if energy_ideal else np.nan,
    }


IRRADIANCE_COLUMNS = ['GHI (W/m^2)', 'DNI (W/m^2)', 'DHI (W/m^2)']


def analyze_file(fp, site_params, start_date=None, end_date=None):
    df = load_stow_conditions(fp)
    df = filter_date_range(df, start_date, end_date)
    result = {'file': os.path.basename(fp), 'rows': len(df)}
    result.update(trigger_counts(df))
    result.update(time_breakdown(df))

    if all(col in df.columns for col in IRRADIANCE_COLUMNS):
        result.update(irradiance_capture(df, site_params))
    else:
        print(f'  Skipping irradiance capture for {os.path.basename(fp)}: '
              f'missing {[c for c in IRRADIANCE_COLUMNS if c not in df.columns]}')
        result.update({
            'poa_energy_stow_Wh_m2': np.nan,
            'poa_energy_ideal_Wh_m2': np.nan,
            'poa_energy_diff_Wh_m2': np.nan,
            'poa_energy_pct_diff': np.nan,
        })
    return result


def main(pattern=None, output_csv=None, site_key='site_example',
         start_date='2024-04-02', end_date='2024-04-03'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'Output')
    pattern = pattern or os.path.join(output_dir, f'{site_key}_*_stow_conditions.csv')

    files = sorted(glob.glob(pattern))
    if not files:
        print(f'No files matched pattern: {pattern}')
        return None

    site_params = SITE_CONFIGS[site_key]

    if start_date or end_date:
        print(f'Restricting analysis to local dates {start_date} through {end_date}')

    rows = []
    for fp in files:
        print(f'Analyzing {os.path.basename(fp)}...')
        rows.append(analyze_file(fp, site_params, start_date=start_date, end_date=end_date))

    # Transpose so each file is a column and each metric is a row.
    summary = pd.DataFrame(rows).set_index('file').T
    pd.set_option('display.width', 160)
    print('\nInitial Site Stow-Condition Metrics:')
    print(summary.round(3).to_string())

    if output_csv:
        summary.to_csv(output_csv)
        print(f'\nSaved summary to {output_csv}')

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern', help='Glob pattern for stow_conditions CSVs to analyze')
    parser.add_argument('--output-csv', default='year', help='Optional path to save the summary table as CSV')
    parser.add_argument('--site-key', default='site_example',
                         help='Key into sites.py SITE_CONFIGS used for location/mount params')
    parser.add_argument('--start-date', default='',#'2024-04-02',
                         help="Local start date (inclusive), e.g. '2024-04-02'. Pass '' to disable.")
    parser.add_argument('--end-date', default='',#'2024-04-03',
                         help="Local end date (inclusive), e.g. '2024-04-03'. Pass '' to disable.")
    args = parser.parse_args()

    main(pattern=args.pattern, output_csv=args.output_csv, site_key=args.site_key,
         start_date=args.start_date or None, end_date=args.end_date or None)
