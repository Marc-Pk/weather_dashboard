# Weather Dashboard

A simple weather monitoring system using the ENS160+AHT21 sensor. Measurements are saved to a SQLite database and plotted in an interactive dashboard, including weather report data if desired.

## Features
- Real-time display of indoor temperature, humidity, and eCO2 levels
- Integration with outdoor weather data
- Multiple visualization options with adjustable time ranges and granularity
![weather_dashboard_3](https://github.com/user-attachments/assets/8e0bcacc-5f5a-46bb-9f2d-7ec137f032e3)
![weather_dashboard_2](https://github.com/user-attachments/assets/898264f0-1d34-43da-888e-4dfe1d7d7dd6)
![weather_dashboard_1](https://github.com/user-attachments/assets/31d14e7d-6abc-4a5d-8da3-0da8eb30e42c)
## Setup

1. Clone this repository:

2. Install the required Python packages with 

    ```

   pip install -r requirements.txt
   ```

3. Set up your sensors, ensure they're properly connected to your Raspberry Pi and that you have installed the necessary libraries. If you have different sensors, you can try to modify the weather_sensor.py file if they can yield the same variables.

4. Move the ```weather_sensor.py``` file to your Raspberry Pi. The database will be generated in the working directory that is used to call the script.

5. Recommended but not strictly necessary: Set up a cronjob to execute the script at boot.

6. Update the `config.py` file with your specific settings:
   - Set the `DB_PATH` to the location where your sensor data is stored. Personally I connect my Raspberry Pi as a network drive to achieve this, but you could also host the dashboard on the Pi itself of course.
   - Update `LATITUDE` and `LONGITUDE` to the coordinates of your location to get accurate outdoor weather data.

## Usage

1. Start the sensor data collection:
   ```
   python weather_sensor.py
   ```
   This script will run continuously, collecting data from your sensors and storing it in the SQLite database.

2. Launch the dashboard:
   ```
   python weather_dashboard.py
   ```
   The dashboard will be accessible at `http://localhost:8050`.
