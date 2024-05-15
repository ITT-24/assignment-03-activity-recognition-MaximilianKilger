import pandas as pd
import os

NAME = 'max'
ACTION = 'lifting'
NUMBER = -1


# read csv
df = pd.read_csv(os.path.join('data',f'{NAME}-{ACTION}-{NUMBER}.csv'))

print(df["timestamp"])
# convert timestamps to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# do resample (100Hz - one data point every 10ms)
df.set_index('timestamp', inplace=True)
df_resampled = df.resample('10ms').mean() 

# reset index from 1 to n instead of using original indices
df_resampled.reset_index(inplace=True)

# convert back to timestamps 
df_resampled['timestamp'] = (df_resampled['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')

# save resampled data to to csv using 'id' as index column name
df_resampled.index.name = 'id'
df_resampled.to_csv(os.path.join('data',f'{NAME}-{ACTION}-{NUMBER}.csv'), index=True)
