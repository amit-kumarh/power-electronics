import csv
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

R = 5 # Load Resistance
VG = 18 # Input Voltage
Fs = 50500 # Measured frequency
Ts = 1/ Fs
R_SHUNT = 0.05 # from schematic

# read funky Rigol CSV format
def read_rigol_csv(csv_file_name):
    with open(csv_file_name) as f:
        rows = list(csv.reader(f))
        i = 0
        while rows[0][i] != "":
            i = i+1
        numcols = i-2
        t0 = float(rows[1][numcols])
        dT = float(rows[1][numcols+1])

    data = pd.read_csv(csv_file_name, usecols=range(0,numcols), skiprows=[1])
    data['X'] = t0+data['X']*dT
    return data, t0, dT

def import_and_clean(name, cols, filt, filt_len=10):
    """
    Import and clean some data from the Rigol scope

    @param name: name of csv file (assumed to live in `./data`)
    @param cols: what to rename columns to
    @param filt: whether to apply 10 sample rolling mean LPF

    @return pd.df - renamed/filtered dataframe
    """
    data, t0, dT = read_rigol_csv(f"data/{name}.csv")
    if 'CH1' in data:
        data[cols[0]] = data['CH1'].rolling(filt_len).mean() if filt else data['CH1'] # filter/rename data
    if 'CH2' in data:
        data[cols[1]] = data['CH2'].rolling(filt_len).mean() if filt else data['CH2']
    if 'CH3' in data:
        data[cols[2]] = data['CH3'].rolling(filt_len).mean() if filt else data['CH3']
    data['X'] = data['X'].subtract(t0) # start x-axis from 0
    return data, dT

def duty_cycle(df, wave_name, start, end, thresh=0.25):
    """
    Calcluate duty cycle of given column in a df.

    @param df: dataframe containing 'X' time column and waveform data
    @param wave_name: column name of waveform to analyze
    @param start: start time prior to signal being held low
    @param end: end time after signal is low
    @param Ts: switching period
    @param thresh: threshold below which to consider the FET on

    @return np.float: duty cycle
    """
    wave = df[wave_name]
    x = df['X']
    rng = np.where((x > start) & (x < end))[0]
    wave = wave[rng[0]:rng[-1]]
    lows = np.where(wave < thresh)[0]
    dt = np.diff(x)[0] # should be constant
    return dt * len(lows) / Ts

def calc_inductance(df, start, end, V_L):
    """
    Calculate inductance from the inductor shunt voltage

    @param df: dataframe containing 'X' time column and "Vshunt" column
    @param start: start time during current ramp
    @param end: end time during current ramp
    @param Vout: output voltage of converter
    
    @return np.float: inductance in H
    """
    I_calc = df["Vshunt"] / R_SHUNT
    rng = np.where((df["X"] > start) & (df["X"] < end))[0]
    dI = np.polyfit(df["X"][rng[0]:rng[-1]], I_calc[rng[0]:rng[-1]], 1)[0]
    return V_L/dI

def frequency(signal, dt, prominence, distance):
    peaks, _ = find_peaks(signal, prominence=prominence, distance=distance)
    if len(peaks) < 2:
        return None  # Not enough peaks to calculate frequency

    # Calculate the average distance between peaks
    distance = np.mean(np.diff(peaks))
    period = distance * dt

    # Calculate the period and frequency
    return peaks, 1/period

def damping_ratio(signal, peaks, offset=0):
    peak_vals = np.array(signal.iloc[peaks]) - offset
    delta = np.mean(np.log(peak_vals[:-1] / peak_vals[1:]))
    zeta = delta / np.sqrt(4*np.pi**2 + delta**2)
    return zeta