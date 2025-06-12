import time
import cv2
from PIL import Image
import numpy as np
from selenium import webdriver 
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import tkinter as tk  # For getting screen dimensions


class WebDriver():
    def __init__(
            self, 
            driver_dir, 
            runner_url, 
            id, 
            window_width=800, 
            window_height=600, 
            window_position='center', 
            frameless=False
        ):
        service = Service(driver_dir)
        options = webdriver.ChromeOptions()
        
        # Get screen dimensions
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        
        # Calculate window position
        if window_position == 'center':
            position_x = (screen_width - window_width) // 2
            position_y = (screen_height - window_height) // 2
        elif window_position == 'top-left':
            position_x = 0
            position_y = 0
        elif window_position == 'top-right':
            position_x = screen_width - window_width
            position_y = 0
        elif window_position == 'bottom-left':
            position_x = 0
            position_y = screen_height - window_height
        elif window_position == 'bottom-right':
            position_x = screen_width - window_width
            position_y = screen_height - window_height
        
        options.add_argument(f'--window-size={window_width},{window_height}')
        options.add_argument(f'--window-position={position_x},{position_y}')
        # Add frameless window option
        if frameless:
            options.add_argument('--app=' + runner_url)  # This creates a frameless window
        
        self.driver = webdriver.Chrome(service=service, options=options)
        if not frameless:
            self.driver.get(runner_url)
        self.handler = self.driver.find_element(By.ID, id)
        
        # Ensure window size and position are correctly set
        if not frameless:
            self.driver.set_window_size(window_width, window_height)
            self.driver.set_window_position(position_x, position_y)

    def get_byte_image(self):
        return self.driver.get_screenshot_as_png()
    
    def quit(self):
        self.driver.quit()

class Runner_Env():
    def __init__(
            self, 
            web_driver, 
            number_of_states, 
            ending_bounding_box=None, 
            track_bounding_box=None, 
            state_grid_rows=2, 
            state_grid_cols=10, 
            state_binary_threshold=0.2,
            is_done_threshold=0.6
        ):
        self.number_of_state = number_of_states
        self.web_driver = web_driver
        self.ending_bounding_box = ending_bounding_box  # Default values for "G" letter
        self.track_bounding_box = track_bounding_box  # Default values for track area
        self.state_grid_rows = state_grid_rows
        self.state_grid_cols = state_grid_cols
        self.state_binary_threshold = state_binary_threshold
        self.is_done_threshold = is_done_threshold
    
    def get_game_screenshot(self):
        og_image = self.web_driver.get_byte_image()
        og_image = np.fromstring(og_image, np.uint8)
        og_image = cv2.imdecode(og_image, cv2.IMREAD_COLOR)  # whole image
        return og_image
    
    def get_binary_image(self, image):
        binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary_image = cv2.adaptiveThreshold(
            binary_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  
            11,
            2
        )
        return binary_image
    
    def get_image(self, image_wait_time=0.1):
        image_1 = self.get_game_screenshot()
        binary_image_1 = self.get_binary_image(image_1)
        time.sleep(image_wait_time)
        image_2 = self.get_game_screenshot()
        binary_image_2 = self.get_binary_image(image_2)
        return binary_image_1, binary_image_2
    
    def get_track(self, image):
        # Extract track area using track_bounding_box
        i, j, height, width = self.track_bounding_box
        # Handle -1 values
        if width == -1:
            width = image.shape[1] - j
        if height == -1:
            height = image.shape[0] - i
        track = image[i:i+height, j:j+width]
        return track
    
    def get_ending(self, image):
        # Extract end image using ending_bounding_box
        i, j, height, width = self.ending_bounding_box
        # Handle -1 values
        if width == -1:
            width = image.shape[1] - j
        if height == -1:
            height = image.shape[0] - i
        end_image = image[i:i+height, j:j+width]
        return end_image
    
    def is_done(self, image):
        end_image = self.get_ending(image)
        overall = end_image.shape[0] * end_image.shape[1]
        return np.count_nonzero(end_image) > overall * self.is_done_threshold
    
    def split_states(self, track_image):
        # Calculate the size of each state in the grid
        states_grid_width = track_image.shape[1] // self.state_grid_cols
        states_grid_height = track_image.shape[0] // self.state_grid_rows
        
        # Calculate cuts to make the image divisible by state_grid_width
        top_cut = (track_image.shape[1] - (states_grid_width * self.state_grid_cols)) // 2
        bottom_cut = (track_image.shape[1] - (states_grid_width * self.state_grid_cols)) - top_cut
        left_cut = (track_image.shape[0] - (states_grid_height * self.state_grid_rows)) // 2
        right_cut = (track_image.shape[0] - (states_grid_height * self.state_grid_rows)) - left_cut
        
        # Truncate the image to make it divisible by state_grid_width
        truncated_track_image = track_image[left_cut:-right_cut if right_cut > 0 else None, 
                                          top_cut:-bottom_cut if bottom_cut > 0 else None]
        
        # Split the image into a grid of states
        # First split into rows, then split each row into columns
        rows = np.array_split(truncated_track_image, self.state_grid_rows, axis=0)
        states_grid = np.array([np.array_split(row, self.state_grid_cols, axis=1) for row in rows])
        
        return states_grid
    
    def compute_state_value_binary(self, state_grid):
        """
        Compute the state value of the state grid.
        The state value is the number of non-zero pixels in the state grid.
        The binary is 1 if the state value is less than the state_binary_threshold, otherwise 0.
        Args:
            state_grid: array with shape [state_grid_height, state_grid_width, state_size_per_layer, state_size_per_layer]
        Returns:
            binary: array with shape [state_grid_height, state_grid_width]
        """
        state_real_value = self.compute_state_value_real(state_grid)
        state_grid_height, state_grid_width = state_grid.shape[2:]
        threshold = state_grid_height * state_grid_width * self.state_binary_threshold
        # print(f"state_grid_height: {state_grid_height}")
        # print(f"state_grid_width: {state_grid_width}")
        # print(f"state_binary_threshold: {self.state_binary_threshold}")
        # print(f"threshold: {threshold}")
        binary = (state_real_value > threshold).astype(np.uint8)
        return binary
    
    def compute_state_value_between_01(self, state_grid):
        """
        Compute the state value of the state grid.
        The state value is the number of non-zero pixels in the state grid.
        The binary is 1 if the state value is less than the state_binary_threshold, otherwise 0.
        Args:
            state_grid: array with shape [state_grid_height, state_grid_width, state_size_per_layer, state_size_per_layer]
        Returns:
            binary: array with shape [state_grid_height, state_grid_width]
        """
        state_real_value = self.compute_state_value_real(state_grid)
        state_grid_height, state_grid_width = state_grid.shape[2:]
        overall_pixel = state_grid_height * state_grid_width
        # print(f"state_grid_height: {state_grid_height}")
        # print(f"state_grid_width: {state_grid_width}")
        # print(f"state_binary_threshold: {self.state_binary_threshold}")
        # print(f"threshold: {threshold}")
        return state_real_value/overall_pixel
    
    def compute_state_value_real(self, state_grid):
        """
        Compute the state value of the state grid.
        The state value is the number of non-zero pixels in the state grid.
        Args:
            state_grid: array with shape [state_grid_height, state_grid_width, state_size_per_layer, state_size_per_layer]
        Returns:
            state_value: array with shape [state_grid_height, state_grid_width]
        """
        state_value = np.count_nonzero(state_grid, axis=(2,3), keepdims=False)
        return state_value
    
    def get_states(self, binary_image_1, binary_image_2):
        track_image_1 = self.get_track(binary_image_1)       
        states_grid_1 = self.split_states(track_image_1)
        state_value_1 = self.compute_state_value_binary(states_grid_1)
        # state_value = self.compute_state_value_between_01(states_grid)
        state_value_1 = state_value_1.T.flatten()

        track_image_2 = self.get_track(binary_image_2)       
        states_grid_2 = self.split_states(track_image_2)
        state_value_2 = self.compute_state_value_binary(states_grid_2)
        # state_value = self.compute_state_value_between_01(states_grid)
        state_value_2 = state_value_2.T.flatten()
        return state_value_1, state_value_2
    
    def start(self):
        self.web_driver.handler.send_keys(Keys.SPACE)

    def wait(self, sleep_time):
        time.sleep(sleep_time)

    def get_time(self):
        return time.time()

    def env_destroy(self):
        self.web_driver.quit()