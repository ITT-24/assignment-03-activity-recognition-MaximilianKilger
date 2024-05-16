# this program gathers sensor data
from DIPPID import SensorUDP
import numpy as np
import pandas as pd
import time
from threading import Thread
import seaborn as sns
import matplotlib.pyplot as plt
from pynput.keyboard import Listener, Key
import os


PORT = 5700
sensor = SensorUDP(PORT)


#dataframe storing sensor measurements
cols = [
    #"id",
    "timestamp",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z"
]
data = pd.DataFrame(columns=cols)

# Name of the activity recorded
ACTIVITY = "rowing"
LOG_NUMBER = 5


#SAMPLE_RATE = 200

global keep_recording
keep_recording = False

# polls sensor for data
def get_sensor_data():
    if sensor.has_capability('accelerometer') and sensor.has_capability('gyroscope'):
        acc_x = sensor.get_value('accelerometer')['x']
        acc_y = sensor.get_value('accelerometer')['y']
        acc_z = sensor.get_value('accelerometer')['z']
        gyro_x = sensor.get_value('gyroscope')['x']
        gyro_y = sensor.get_value('gyroscope')['y']
        gyro_z = sensor.get_value('gyroscope')['z']

        return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    return None

# takes values from get_sensor_data and appends them to dataframe, also logs id and timestamp 
def record_data():
    print("Recording...")
    global keep_recording
    global data
    index = 0
    while keep_recording:
        ts = time.time()
        values = get_sensor_data()
        if values != None:
            #row = pd.DataFrame(columns=cols)
            #row['id'] = index
            #row['timestamp'] = ts
            acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = values
            #print('.'.join([ str(acc_x), str(acc_y), str(acc_z), str(gyro_x), str(gyro_y), str(gyro_z)]))
            #row['acc_x'] = acc_x
            #row['acc_y'] = acc_y
            #row['acc_z'] = acc_z
            #row['gyro_x'] = gyro_x
            #row['gyro_y'] = gyro_y
            #row['gyro_z'] = gyro_z
            difference = 0
            
            #if len(data) > 0:
            #    difference = ts - data.iloc[-1]["timestamp"]
            row = pd.DataFrame([[ ts, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]], columns=cols)
            if len(data) == 0:
                data = row
            else:
                row1 = data.iloc[len(data)-1]
                row2 = row.iloc[0]
                hasduplicates = (row1['acc_x'] == row2['acc_x'] and
                                row1['acc_y'] == row2['acc_y'] and
                                row1['acc_z'] == row2['acc_z'] and
                                row1['gyro_x'] == row2['gyro_x'] and
                                row1['gyro_y'] == row2['gyro_y'] and
                                row1['gyro_z'] == row2['gyro_z']) 
                if not hasduplicates:
                    data = pd.concat([data, row])
                else:
                    index -= 1
            #print(data)
            index += 1

        # calculate time that thread has to sleep so that measurement happens at (roughly) regular intervals
        #delta_time = time.time()-ts
        #if delta_time < 1/SAMPLE_RATE:
        #    time.sleep(1/SAMPLE_RATE - delta_time)
        time.sleep(0.003)
    
    print("Recording thread finished")
    print(data)
    return

#showplot = False
#starts measurement thread
def start_recording():
    global keep_recording
    global showplot
    keep_recording = True
    print("Starting recording_thread")
    recording_thread = Thread(target=record_data)
    recording_thread.start()
#    while True:
#        plt.show()

def on_press(key):
    global keep_recording
    global showplot
    print(f"Key pressed {str(key)}")
    if key == Key.space:
        if keep_recording == False:
            start_recording()
        else:
            keep_recording = False

#    if key.char == 'p':
#        sns.lineplot(data,y='acc_x', x='timestamp')
#        showplot = True

    if str(key) == '\'s\'':
        print("Saving...")
        global data
        #data = data.drop('index')

        # AS: hard coded path?
        filepath = os.path.join("data", f"andi-{ACTIVITY}-{LOG_NUMBER}.csv")
        data.reset_index(inplace=True)
        data.index.name = 'id'
        data.drop('index', axis=1, inplace=True)
        print(data)
        data.to_csv(filepath)

    if str(key) == '\'q\'':
        os._exit(0)
            
        
    
def on_release(key):
    pass

with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()


        
    


