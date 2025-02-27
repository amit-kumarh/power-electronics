import csv
import pandas as pd
import numpy as np

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

def import_and_clean(name, cols, filt):
    """
    Import and clean some data from the Rigol scope

    @param name: name of csv file (assumed to live in `./data`)
    @param cols: what to rename columns to
    @param filt: whether to apply 10 sample rolling mean LPF

    @return pd.df - renamed/filtered dataframe
    """
    data, t0, dT = read_rigol_csv(f"data/{name}.csv")
    data[cols[0]] = data['CH1'].rolling(10).mean() if filt else data['CH1'] # filter/rename data
    data[cols[1]] = data['CH2'].rolling(10).mean() if filt else data['CH2']
    data['X'] = data['X'].subtract(t0) # start x-axis from 0
    return data, dT

def duty_cycle(df, wave_name, start, end, Ts, thresh=0.25):
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

def calc_inductance(df, start, end, Vout):
    """
    Calculate inductance from the inductor shunt voltage

    @param df: dataframe containing 'X' time column and "Vshunt" column
    @param start: start time during current ramp
    @param end: end time during current ramp
    @param Vout: output voltage of converter
    
    @return np.float: inductance in H
    """
    I_calc = df["Vshunt"] / R_SHUNT
    V_L = VG - Vout
    rng = np.where((df["X"] > start) & (df["X"] < end))[0]
    dI = np.polyfit(df["X"][rng[0]:rng[-1]], I_calc[rng[0]:rng[-1]], 1)[0]
    return V_L/dI



