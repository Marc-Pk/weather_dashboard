# Weather Dashboard

A simple weather monitoring system using the ENS160+AHT21 sensor. Measured data is saved either to a DynamoDB on AWS or a SQLite database. The data can then be plotted in an interactive dashboard, including weather report data using the open-meteo API if desired.

## Features
- Real-time display of measured temperature, humidity, and eCO2 levels
- save data locally or on AWS
- Integration with API-based weather data
- Multiple visualization options with adjustable time ranges and granularity
![weather_dashboard_3](https://github.com/user-attachments/assets/8e0bcacc-5f5a-46bb-9f2d-7ec137f032e3)
![weather_dashboard_2](https://github.com/user-attachments/assets/898264f0-1d34-43da-888e-4dfe1d7d7dd6)
![weather_dashboard_1](https://github.com/user-attachments/assets/31d14e7d-6abc-4a5d-8da3-0da8eb30e42c)
## Setup

First of all, clone this repository.

### On your Raspberry Pi:

1. Set up your sensors and ensure they're properly connected. If you have different sensors, you can modify the ```weather_sensor.py``` file given that they can measure the same variables.

2. Move the ```sensor``` folder to your RPI and install the required Python packages with 

   ```
   pip install -r requirements_sensor.txt
   ```

3. Set the ```DB_TYPE``` variable in the ```weather_sensor.py``` file to "LOCAL" if you want to use a local database, which will be generated in the working directory that is used to call the script. Leave it at ```AWS``` to use a DynamoDB database that needs to have a partition key called "Time". Make sure that your AWS credentials are available on your RPI.

4. Start the sensor script:
   ```
   python weather_sensor.py
   ```
   The script will run continuously, collecting data from your sensors and storing it in the selected database.

5. Recommended but not strictly necessary: Set up a cronjob using ```crontab -e``` to execute the ```weather_sensor.py``` script at boot. For example:
   ```
   @reboot cd /<path>/weather && sudo python weather_sensor.py &
   ```

### Setting up the dashboard
1. In the dashboard folder, install the required Python packages with 

   ```
   pip install -r requirements_dashboard.txt
   ```

2. Modify the config values in the ```weather_dashboard.py``` file to use your local coordinates and database settings.

3. Launch the dashboard:
   ```
   python weather_dashboard.py
   ```
   Alternatively, build your own docker container with the provided dockerfile. The dashboard will be accessible at `http://localhost:8050`.
 
