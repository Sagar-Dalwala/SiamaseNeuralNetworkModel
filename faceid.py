# Import kivy dependencies first

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX dependencies
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import kivy graphics
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies 
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Build App Layout
class CamApp(App):
    
    def build(self):
        # Main layout components
        self.web_cam = Image(size_hint = (1,.8))
        self.button = Button(text = "Verify", on_press = self.verify ,size_hint = (1,.1))
        self.close_button = Button(text="Close App", size_hint=(1, .1), background_color=(1, 0, 0, 1)) 
        self.verification_label = Label(text = "Verification Uninitiated" , size_hint = (1,.1))
        
        # Add components to layout
        layout = BoxLayout(orientation = "vertical")
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.close_button)
        layout.add_widget(self.verification_label)
        
        # Bind the close button to the close_app function
        self.close_button.bind(on_press=self.close_app)
        
        # Load tensorflow keras model
        self.model = tf.keras.models.load_model('siamesemodelv2.h5' , custom_objects={'L1Dist':L1Dist})
        
        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        url = "v1.mp4"
        self.capture.open(url)
        
        Clock.schedule_interval(self.update , 1.0/33.0)
        
        return layout
    
    # Run continuously to get frames from the camera feed
    def update(self, *args):
        
        # Read frame from opencv
        ret , frame = self.capture.read()
        
        #cut down frame to 400x400 pixels 
        # frame = frame[50:50+500,720:720+500,:]
        
        # Filp horizontal and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size = (frame.shape[1] , frame.shape[0]) ,colorfmt='bgr')
        img_texture.blit_buffer(buf , colorfmt='bgr' , bufferfmt = 'ubyte')
        self.web_cam.texture = img_texture
    
    # Load image from file path and convert to 100x100x3
    def preprocess(self,file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image
        img = tf.io.decode_jpeg(byte_img)
        
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img , (100,100))
        # Scale image to be between 0 and 1
        img = img / 255.0
        
        # Return image
        return img
    
    # Verify function - to compare the image with the database
    def verify(self, *args):
        # Specify thresholds
        detection_threshold = 0.5
        verification_threshold = 0.8
        
        # Capture input image from webcam
        SAVE_PATH = os.path.join('application_data' , 'input_image' , 'input_image.jpg')
        ret , frame = self.capture.read()
        # frame = frame[50:50+500,720:720+500,:]
        cv2.imwrite(SAVE_PATH , frame)
        
        # Build results array
        results = []
        for image in os.listdir(os.path.join('application_data' , 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data' , 'input_image' , 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data' , 'verification_images',image))
            
            # Make Predictions
            result = self.model.predict(list(np.expand_dims([input_img , validation_img] , axis=1)))
            print(result[0][0])
            results.append(result)
        
        # Detection Threshold :  Metric above which a predicition is considered positive
        detection = np.sum(np.array(results) > detection_threshold)
        
        # Verification Threshold : Proportion of positive predictions / total positive samples
        verification = detection / len(os.listdir(os.path.join('application_data','verification_images')))
        verified = verification > verification_threshold
        
        # Set Verification Text
        self.verification_label.text = "Verification Successful" if verified == True else "Verification Failed!!!"
        
        # Log out details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)
        # Logger.info(np.sum(np.array(results) > 0.2))
        # Logger.info(np.sum(np.array(results) > 0.4))
        # Logger.info(np.sum(np.array(results) > 0.5))
        # Logger.info(np.sum(np.array(results) > 0.8))
        
        return results , verified
    
    # Function to close the application
    def close_app(self, instance):
        App.get_running_app().stop()
        # Release the video capture device
        self.capture.release()
    
if __name__ == '__main__':
    CamApp().run()