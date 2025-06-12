#!/bin/bash

# Run the image test script with the specified parameters
python main.py \
    --mode image \
    --image_wait_time 0.00 \
    --number_of_actions 3 \
    --chrome_driver_path "/home/newton/chrome_driver/chromedriver-linux64/chromedriver" \
    --game_url "http://localhost:8000/" \
    --window_width 600 \
    --window_height 300 \
    --window_position center \
    --frameless \
    --ending_i 90 \
    --ending_j 280 \
    --ending_height 35 \
    --ending_width 40 \
    --track_i 100 \
    --track_j 70 \
    --track_height 45 \
    --track_width 300 \
    --state_grid_rows 1 \
    --state_grid_cols 12 \
    --state_binary_threshold 0.05 \
    --is_done_threshold 0.5 \