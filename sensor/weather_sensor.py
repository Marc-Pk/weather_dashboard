#!/usr/bin/python
import logging
import board
import adafruit_ens160
import adafruit_ahtx0
from decimal import Decimal
from time import sleep
from statistics import median
from datetime import datetime

# Setup for logging
logging.basicConfig(filename="weather_sensor.log", level=logging.ERROR, datefmt="%Y-%m-%d %H:%M:%S", format="#######\n%(asctime)s\n%(message)s\n\n")

# Set the database type to either "LOCAL" or "AWS"
DB_TYPE = "AWS"

# Name of your DynamoDB database
DB_PATH = 'sensor-data'

# Initialize the database/API connection
if DB_TYPE == "LOCAL":
    import sqlite3
    conn = sqlite3.connect('sensor_data.db')
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data
        (Time TIMESTAMP, Humidity REAL, Temperature REAL, TVOC INTEGER, eCO2 INTEGER)
        ''')
    conn.commit()
    
else:
    import boto3
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

    table_name = DB_PATH
    table = dynamodb.Table(table_name)

# if you want to use different sensors, adapt this block, the imports and the first block in the while loop
i2c = board.I2C()
ens = adafruit_ens160.ENS160(i2c)
aht = adafruit_ahtx0.AHTx0(i2c)

# Initialize data list
data = []

# Main loop
while True:
    sleep(5)
    
    temperature = aht.temperature - 4.5
    humidity = aht.relative_humidity
    ens.temperature_compensation = temperature
    ens.humidity_compensation = humidity
    tvoc = ens.TVOC
    eCO2 = ens.eCO2
   
    current_time = datetime.now()
    data.append((current_time, humidity, temperature, tvoc, eCO2))
   
    # Calculate median values once per minute and send them to the database
    if current_time.minute != data[0][0].minute:
        # Calculate median values
        median_data_raw = (data[0][0].replace(second=0, microsecond=0), median([x[1] for x in data]), median([x[2] for x in data]), median([x[3] for x in data]), median([x[4] for x in data]))
        median_data = (median_data_raw[0].strftime('%Y-%m-%d'), median_data_raw[0].strftime('%Y-%m-%d %H:%M:%S'), round(median_data_raw[1], 2), round(median_data_raw[2], 2), round(median_data_raw[3]), round(median_data_raw[4]))

        try:
            # Insert data into the DynamoDB database
            if DB_TYPE == "AWS":
                response = table.put_item(
                    Item={
                        'date': str(median_data[0]),
                        'datetime': str(median_data[1]),
                        'Humidity': Decimal(str(median_data[2])),
                        'Temperature': Decimal(str(median_data[3])),
                        'TVOC': Decimal(str(median_data[4])),
                        'eCO2': Decimal(str(median_data[5]))
                    }
                )       

            # Alternatively, insert data into the SQLite database
            else:
                cursor.execute('''INSERT INTO sensor_data (date, datetime, Humidity, Temperature, TVOC, eCO2) VALUES (?, ?, ?, ?, ?, ?)''', median_data)
                conn.commit()
        
            # Clear data
            data = []
        except Exception as e:
            logging.exception(e)