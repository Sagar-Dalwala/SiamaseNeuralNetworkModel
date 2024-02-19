# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2

# Build App Layout
class CamApp(App):
    
    def build(self):
        # Main layout components
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button(text="Verify", size_hint=(1, .1))
        self.close_button = Button(text="Close App", size_hint=(1, .1), background_color=(1, 0, 0, 1))  # Red color for the close button
        self.verification = Label(text="Verification Uninitiated", size_hint=(1, .1))
        
        # Add components to layout
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.close_button)
        layout.add_widget(self.verification)
        
        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        url = "my_face.mp4"  # Corrected the path format
        self.capture.open(url)
        
        # Bind the close button to the close_app function
        self.close_button.bind(on_press=self.close_app)
        
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout
    
    # Run continuously to get frames from the camera feed
    def update(self, *args):
        # Read frame from opencv
        ret, frame = self.capture.read()
        
        # Check if the frame is captured successfully
        if ret:
            # Flip horizontal and convert image to texture
            buf = cv2.flip(frame, 0).tostring()
            img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.web_cam.texture = img_texture
    
    # Function to close the application
    def close_app(self, instance):
        App.get_running_app().stop()
        # Release the video capture device
        self.capture.release()

if __name__ == '__main__':
    CamApp().run()
