#!/bin/bash

python main.py \
    --mode test \
    --epochs 5 \
    --model_path "./runs/20250605_022421/models/my_model_137.pt" \
    --chrome_driver_path "/home/newton/chrome_driver/chromedriver-linux64/chromedriver" \
    --game_url "http://localhost:8000/" \
    --number_of_actions 3 \
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
    --jump_duration 0.5 \
    --crawl_duration 0.3