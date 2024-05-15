# this program recognizes activities
import numpy as np
import pandas as pd
import os
from scipy.fft import fft
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# shamelessly stolen from my 2nd assignment.
# given a ndarray (float) of samples, finds the most energetic frequency in the fourier transform of the signal
def find_prevalent_frequency(data:np.ndarray, sampling_rate:int, filter_low:float=None, filter_high:float=None) -> float:
    use_filter = filter_low != None and filter_high != None
    wd = np.zeros((len(data)))
    if use_filter:
        sos = signal.butter(5, [filter_low, filter_high], 'bandpass', fs=sampling_rate, output='sos') #apply bandpass filter
        filtered_data = signal.sosfiltfilt(sos, data)
        wd = np.hamming(len(filtered_data)) * filtered_data
    else:
        wd = np.hamming(len(data)) * data #apply hamming window
    fourier = np.fft.fft(wd)
    if np.isnan(np.sum(fourier)):
        return None
    frequencies = np.fft.fftfreq(len(wd), 1/sampling_rate)
    
    highest_frequency = np.max(frequencies[np.where(np.abs(fourier) == np.max(np.abs(fourier)))])
    return abs(highest_frequency)

class ActivityRecognizer():
    DATA_DIRECTORY = "data"
    ACTIVITIES = ["lifting", "rowing", "jumpingjack", "running"]

    COLS_RAW = ["id","timestamp","acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z", "activity", "source"]
    
    SAMPLING_RATE = 0.01

    def __init__(self):
        self.df_raw = self.load_dataset()
        self.dataset_x, self.dataset_y = self.extract_features(self.df_raw, fit_scaler=True)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.dataset_x, self.dataset_y)
        


    def load_dataset(self)->pd.DataFrame:
        df_raw = pd.DataFrame(columns=self.COLS_RAW)
        # iterate through everything in the data directory
        for data_source in os.listdir(self.DATA_DIRECTORY):
            for data_sheet in os.listdir(os.path.join(self.DATA_DIRECTORY, data_source)):
                activity = ""
                for potential_activity in self.ACTIVITIES:
                    if potential_activity in data_sheet:
                        activity = potential_activity
                        break
                # discard non-csvs
                if not data_sheet.endswith('csv'):
                    continue
                #load csv
                df = pd.read_csv(os.path.join(self.DATA_DIRECTORY, data_source, data_sheet))

                df["activity"] = [activity] * df.shape[0]
                df["source"] = [data_sheet.split('.')[0]] * df.shape[0]
                
                #concatenate
                if df_raw.shape[0] == 0:
                    df_raw = df
                else:
                    df_raw  = pd.concat((df_raw,df), ignore_index=True)

        return df_raw
    
    def to_frequency_domain (self, raw_data:pd.DataFrame)->pd.DataFrame:
        signal_data_columns = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"] # which columns in raw data can be converted to frequency domain
        
        data_freq_list = []
        row = []
        row_complete = True
        for signal_column in signal_data_columns:
            # find maximum frequency of the movement on a given axis. 
            # there may be better ways of aggregating over time-series data, but I can't think of any at the moment.
            prevalent_freq = find_prevalent_frequency(raw_data[signal_column].values, self.SAMPLING_RATE)
            if prevalent_freq != None:
                row.append(prevalent_freq)
            else:
                row_complete = False
        if not row_complete:
            return None
        else:
            data_freq = pd.DataFrame(np.array(data_freq_list), columns=['acc_x_freq', 'acc_y_freq', 'acc_z_freq', 'gyro_x_freq', 'gyro_y_freq' ,'gyro_z_freq'])
            return data_freq

    
    def to_frequency_domain_training_data (self, raw_data:pd.DataFrame):
        signal_data_columns = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"] # which columns in raw data can be converted to frequency domain
        
        data_freq_list = []
        # process each individually captured series individually
        for data_source in np.unique(raw_data['source']):
            batch = raw_data[raw_data['source'] == data_source]
            row = []
            row_complete = True
            for signal_column in signal_data_columns:
                # find maximum frequency of the movement on a given axis. 
                # there may be better ways of aggregating over time-series data, but I can't think of any at the moment.
                prevalent_freq = find_prevalent_frequency(batch[signal_column].values, self.SAMPLING_RATE)
                if prevalent_freq != None:
                    row.append(prevalent_freq)
                else:
                    row_complete = False
            row.append(batch['activity'].mode()[0])
            if row_complete:
                data_freq_list.append(row)
        data_freq = pd.DataFrame(np.array(data_freq_list), columns=['acc_x_freq', 'acc_y_freq', 'acc_z_freq', 'gyro_x_freq', 'gyro_y_freq' ,'gyro_z_freq', 'activity'])
        return data_freq
    
    def extract_features (self, raw_data, with_training_data = False) -> pd.DataFrame:
        data_freq = None
        if with_training_data:
            data_freq = self.to_frequency_domain_training_data
        else:
            data_freq = self.to_frequency_domain(raw_data)
        if data_freq == None:
            return None
        
        if with_training_data:
            x = data_freq['acc_x_freq', 'acc_y_freq', 'acc_z_freq', 'gyro_x_freq', 'gyro_y_freq' ,'gyro_z_freq']
            y = data_freq['activity']
            self.scaler = MinMaxScaler()
            data_scaled = self.scaler.fit_transform(x, y)
            return data_scaled, y
        else:
            data_scaled = self.scaler.transform(data_freq['acc_x_freq', 'acc_y_freq', 'acc_z_freq', 'gyro_x_freq', 'gyro_y_freq' ,'gyro_z_freq'])
            return data_scaled



                    

                



if __name__ == "__main__":
    ar = ActivityRecognizer()
                