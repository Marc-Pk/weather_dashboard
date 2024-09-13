#!/usr/bin/python
import board
import adafruit_ens160
import adafruit_ahtx0
import sqlite3
from time import sleep
from statistics import median
from datetime import datetime

# Initialize SQLite database
conn = sqlite3.connect('weather_data.db')
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS weather_data
    (Time TIMESTAMP, Humidity REAL, Temperature REAL, TVOC INTEGER, eCO2 INTEGER)
    ''')
conn.commit()

i2c = board.I2C()
ens = adafruit_ens160.ENS160(i2c)
aht = adafruit_ahtx0.AHTx0(i2c)

# Initialize data list
data = []

# Main loop
while True:
    sleep(5)
    temperature = aht.temperature
    humidity = aht.relative_humidity
    ens.temperature_compensation = temperature - 4
    ens.humidity_compensation = humidity + 9
   
    current_time = datetime.now()
    data.append((current_time, humidity, temperature, ens.TVOC, ens.eCO2))
   
    # Calculate median values once per minute and send them to the database
    if current_time.minute != data[0][0].minute:
        median_data = (data[0][0].replace(second=0, microsecond=0), median([x[1] for x in data]), median([x[2] for x in data]), median([x[3] for x in data]), median([x[4] for x in data]))

        median_data = (median_data[0].strftime('%Y-%m-%d %H:%M:%S'), round(median_data[1], 2), round(median_data[2], 2), round(median_data[3]), round(median_data[4]))
       
        # Insert data into SQLite database
        cursor.execute('''INSERT INTO weather_data (Time, Humidity, Temperature, TVOC, eCO2) VALUES (?, ?, ?, ?, ?)''', median_data)
        conn.commit()
       
        # Clear data
        data = []