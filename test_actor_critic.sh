#!/bin/bash

# Actor-Critic Test Script
python main.py \
    --mode test \
    --model_type actor_critic \
    --test_epochs 5 \
    --model_path "./runs/20250916_085941/models/best_actor_model_at_train_round_9_epoch_60.pt" \
    --game_url "http://localhost:8000/" \
    --number_of_actions 2 \
    --image_wait_time 0.0 \
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
    --jump_duration 0.5
