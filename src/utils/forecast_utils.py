
from pydantic import BaseModel, Field, field_validator
from typing import List
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- Validation Schema ---
class ForecastOutput(BaseModel):
    analysis: str = Field(description="Freeform string analysis of the trend.")
    forecast: List[float] = Field(description="List of predicted OT values.")

    @field_validator('forecast')
    @classmethod
    def check_horizon(cls, v: List[float]):
        return v

# --- Dynamic Prompt Builder (Updated for Autoregressive) ---
def get_user_prompt(embedding_type, lookback, horizon, historical_data=None):
    prompt_intro = "You are analyzing Oil Temperature (OT) from an electricity transformer.\n"
    
    if embedding_type == "v":
        context = f"Attached is a plot showing the last {lookback} hours of data."
    elif embedding_type == "d":
        context = f"Historical data for the last {lookback} hours (Celsius): {historical_data}"
    else:  # dv
        context = (f"Attached is a plot showing the last {lookback} hours. "
                   f"The raw values for those hours are: {historical_data}. ")

    instructions = (
        f"\n\n### TASK\n"
        f"Forecast the next hour of OT values (a single value).\n"
        f"### CONSTRAINTS\n"
        f"- Your output should strictly follow the format: 'I predict the next hour will be <value>'\n"
        f"### OUTPUT Example\n"
        f"I predict the next hour will be 37.6"
    )
    return prompt_intro + context + instructions

def build_forecast_prompt(row_idx, current_X_scaled, scaler, L, H, embedding_type, split_name, PromptClass):
    # current_X_scaled is the historical window PLUS any autoregressive steps already predicted
    X_orig = scaler.inverse_transform(np.array(current_X_scaled).reshape(-1, 1)).flatten().tolist()
    X_rounded = [round(x, 2) for x in X_orig[-L:]] # Take only the last 'L' values
    
    # Handle Image generation
    img_path = None
    if "v" in embedding_type:
        save_path = f"./data/images/etth1/{split_name}/{row_idx}_step_{len(current_X_scaled)}.png"
        # We plot the current window known to the model
        img_path = plot_forecasting_question(np.array(current_X_scaled[-L:]), None, scaler, L, H, save_path)
    
    # Build Text
    hist_data = X_rounded if "d" in embedding_type else None
    user_text = get_user_prompt(embedding_type, L, H, hist_data)
    
    return PromptClass(user_text=user_text, image_path=img_path)# --- Utility Functions ---
def create_univariate_windows(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : (i + lookback)])
        y.append(data[(i + lookback) : (i + lookback + horizon)])
    return np.array(X), np.array(y)

def plot_forecasting_question(X, y, scaler, L, H, save_path, recreate=False):
    if os.path.exists(save_path) and not recreate: 
        return save_path
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Inverse transform to show actual Celsius values
    X_orig = scaler.inverse_transform(X.reshape(-1, 1)).flatten()
    
    plt.figure(figsize=(12, 5)) # Slightly wider for better clarity
    
    # 1. Plot historical data
    plt.plot(np.arange(0, L), X_orig, color='blue', linewidth=2, label=f'Historical OT (L={L}h)')
    
    # 2. Add visual indicator for the Forecast Horizon
    plt.axvline(x=L, color='gray', linestyle='--', linewidth=1.5)
    plt.fill_between(np.arange(L, L + H), plt.ylim()[0], plt.ylim()[1], 
                     color='red', alpha=0.05, label=f'Horizon (H={H}h)')
    
    # 3. Add large Question Mark in the target area
    # Note: Using fixed coordinates or relative to y-axis limits for visibility
    y_mean = np.mean(X_orig)
    plt.text(L + (H / 2), y_mean, '?', fontsize=60, color='red', 
             ha='center', va='center', fontweight='bold')
    
    # 4. Labeling (Title and Axes)
    plt.title(f"What is the next {H} hour(s) of Oil Temperature?", fontsize=14)
    plt.xlabel("Time (Hours)", fontsize=12)
    plt.ylabel("Oil Temperature (°C)", fontsize=12)
    
    # 5. Legend and Grid
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150) # Higher DPI for better vision processing
    plt.close()
    
    return save_path