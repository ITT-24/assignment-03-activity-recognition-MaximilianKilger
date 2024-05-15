# this program visualizes activities with pyglet

from activityrecognizer import ActivityRecognizer
import pyglet
from DIPPID import SensorUDP
import time
from threading import Thread
import numpy as np
import pandas as pd
import os

WIN_WIDTH = 600
WIN_HEIGHT = 450

ACTIVITY_SPRITE_ANCHOR_X = WIN_WIDTH / 2
ACTIVITY_SPRITE_ANCHOR_Y = 150

LITTLE_LABEL_ANCHOR_X = WIN_WIDTH / 2 
LITTLE_LABEL_ANCHOR_Y = 400
BIG_LABEL_ANCHOR_X = WIN_WIDTH / 2 
BIG_LABEL_ANCHOR_Y = 350

IMAGE_SCALE = 0.2

TIMEFRAME_FOR_PREDICTION_SEC = 5
UPDATE_INTERVAL_SEC = 1

PORT = 5700

window  = pyglet.window.Window(WIN_WIDTH, WIN_HEIGHT)


class FitnessTrainer():

    def __init__(self):
        self.sensor_data = []
        self.columns = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]
        self.sensor = SensorUDP(PORT)
        self.predicted_activity = None
        self.activity_recognizer = ActivityRecognizer()

        self.setup_ui()

        self.running = True

        self.sensor_thread = Thread(target=self.read_sensor)
        self.sensor_thread.start()

        self.prediction_thread = Thread(target=self.predict_activity)
        self.prediction_thread.start()

    def setup_ui (self):
        self.background = pyglet.shapes.Rectangle(0,0,WIN_WIDTH, WIN_HEIGHT, (250,255,255))
        self.images = {}
        self.images["rowing"] = pyglet.image.load(os.path.join('img','rowing_2.png'))
        self.images["lifting"] = pyglet.image.load(os.path.join('img','lifting_2.png'))
        self.images["running"] = pyglet.image.load(os.path.join('img','running_2.png'))
        self.images["jumpingjack"] = pyglet.image.load(os.path.join('img','jumpingjack_2.png'))

        self.activity_sprite = pyglet.sprite.Sprite(self.images["rowing"],-30000, -30000)

        self.little_label = pyglet.text.Label("", 'Arial', font_size=15,x=LITTLE_LABEL_ANCHOR_X, y=LITTLE_LABEL_ANCHOR_Y, anchor_x='center', anchor_y='center', align='center', color=(0,0,0,255))

        self.big_label = pyglet.text.Label("", 'Arial', font_size=25,x=BIG_LABEL_ANCHOR_X, y=BIG_LABEL_ANCHOR_Y, anchor_x='center', anchor_y='center', align='center', color=(180,10,0,255))
        self.loading_label = pyglet.text.Label('Loading...', 'Arial', font_size=25,x=BIG_LABEL_ANCHOR_X, y=BIG_LABEL_ANCHOR_Y, anchor_x='center', anchor_y='center', align='center', color=(180,10,0,255))
        

    def update_ui(self):
        if self.predicted_activity != None:
            new_image:pyglet.image.AbstractImage = self.images[self.predicted_activity]
            self.activity_sprite.image = new_image
            self.activity_sprite.width = new_image.width * IMAGE_SCALE
            self.activity_sprite.height = new_image.height * IMAGE_SCALE
            self.activity_sprite.x = ACTIVITY_SPRITE_ANCHOR_X - new_image.width*IMAGE_SCALE/2
            self.activity_sprite.y = ACTIVITY_SPRITE_ANCHOR_Y - new_image.height*IMAGE_SCALE/2

            self.little_label.text = "I predict that you are"

            if self.predicted_activity == 'rowing':
                self.big_label.text = "ROWING"
            elif self.predicted_activity == 'lifting':
                self.big_label.text = "LIFTING"
            elif self.predicted_activity == 'running':
                self.big_label.text = "RUNNING"
            elif self.predicted_activity == 'jumpingjack':
                self.big_label.text = "DOING JUMPINGJACKS"

            self.loading_label.text = ''
        else:
            self.loading_label.text = 'Loading...'

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
            if observation != None:
                self.sensor_data.append(observation)
            time.sleep(1*rate)

    def predict_activity(self):
        num_samples_required = int(TIMEFRAME_FOR_PREDICTION_SEC / self.activity_recognizer.SAMPLING_RATE)
        while self.running:
            if len(self.sensor_data) >= num_samples_required:
                subset = self.sensor_data[-num_samples_required-1:-1] # get the last X seconds of sensor data
                data = pd.DataFrame(subset, columns=self.columns)
                self.predicted_activity = self.activity_recognizer.label(data)[0]
                print(self.predicted_activity)
            
            time.sleep(UPDATE_INTERVAL_SEC)

    def render(self):
        self.background.draw()
        self.activity_sprite.draw()
        self.loading_label.draw()
        self.little_label.draw()
        self.big_label.draw()

trainer = FitnessTrainer()



@window.event
def on_draw():
    window.clear()
    trainer.update_ui()
    trainer.render()


@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.Q:
        trainer.running = False
        os._exit(0)


pyglet.app.run()
