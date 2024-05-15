# this program visualizes activities with pyglet

from activityrecognizer import ActivityRecognizer
import pyglet
from DIPPID import SensorUDP
import time
from threading import Thread
import numpy as np
import pandas as pd

WIN_WIDTH = 800
WIN_HEIGHT = 450

TIMEFRAME_FOR_PREDICTION_SEC = 5
UPDATE_INTERVAL_SEC = 1

PORT = 5700

window  = pyglet.window.Window(WIN_WIDTH, WIN_HEIGHT)


class FitnessTrainer():

    def __init__(self):
        self.sensor_data = []
        self.columns = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]
        self.sensor = SensorUDP(PORT)
        self.predicted_activity = ""
        self.activity_recognizer = ActivityRecognizer()
        self.running = True

        self.sensor_thread = Thread(target=self.read_sensor)
        self.sensor_thread.start()

        self.prediction_thread = Thread(target=self.predict_activity)
        self.prediction_thread.start()


    # polls sensor for data
    def get_sensor_data(self):
        if self.sensor.has_capability('accelerometer') and self.sensor.has_capability('gyroscope'):
            acc_x = self.sensor.get_value('accelerometer')['x']
            acc_y = self.sensor.get_value('accelerometer')['y']
            acc_z = self.sensor.get_value('accelerometer')['z']
            gyro_x = self.sensor.get_value('gyroscope')['x']
            gyro_y = self.sensor.get_value('gyroscope')['y']
            gyro_z = self.sensor.get_value('gyroscope')['z']

            return [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        return None
    
    def read_sensor(self):
        rate = self.activity_recognizer.SAMPLING_RATE
        while self.running:
            observation = self.get_sensor_data()
            self.sensor_data.append(observation)
            time.sleep(1*rate)

    def predict_activity(self):
        num_samples_required = int(TIMEFRAME_FOR_PREDICTION_SEC / self.activity_recognizer.SAMPLING_RATE)
        while self.running:
            if len(self.sensor_data) >= num_samples_required:
                subset = self.sensor_data[-num_samples_required-1:-1] # get the last X seconds of sensor data
                data = pd.DataFrame(subset, columns=self.columns)
                self.predicted_activity = self.activity_recognizer.label(data)
                print(self.predicted_activity)
            
            time.sleep(UPDATE_INTERVAL_SEC)
                

trainer = FitnessTrainer()






pyglet.app.run()
