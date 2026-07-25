@echo off
REM This batch file activates the project's virtual environment
REM and runs the traffic data collection script.
REM Task Scheduler will run this file once every hour.

cd /d "D:\Data Science projects\Traffic Forecasting system\karachi-traffic-forecasting"
call venv\Scripts\activate.bat
python src\collect_traffic_data.py
