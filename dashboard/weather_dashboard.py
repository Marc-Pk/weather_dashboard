import warnings
with warnings.catch_warnings(action='ignore'):
    import dash
    import plotly.graph_objects as go
    import dash_bootstrap_components as dbc
    import plotly.io as pio
    import sqlite3
    import openmeteo_requests
    import requests_cache
    import pandas as pd
    import pytz
    import boto3
    import os
    from dash import dcc, html
    from dash.dependencies import Output, Input
    from plotly.subplots import make_subplots
    from flask_caching import Cache
    from retry_requests import retry
    from datetime import datetime, timedelta
    from boto3.dynamodb.conditions import Key

########## Config values ##########
# Either "AWS" if you are using AWS DynamoDB or "LOCAL" for a SQLite database
DB_TYPE = os.getenv("DB_TYPE", "AWS")

# If your DB_TYPE is "AWS", enter the name of your table here
# If your DB_TYPE is "LOCAL", enter the path to your SQLite database here and make sure to include "sensor_data.db" at the end
DB_PATH = os.getenv("DB_PATH", r"sensor-data")

# Coordinates for API weather data
LATITUDE = float(os.getenv("LATITUDE", 49.0094))
LONGITUDE = float(os.getenv("LONGITUDE", 8.4044))

# Timeout for caching of database fetches in seconds
CACHE_TIMEOUT = int(os.getenv("CACHE_TIMEOUT", 300))

class DatabaseHandler:
    def __init__(self):
        if DB_TYPE == "AWS":
            self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
            self.table = self.dynamodb.Table(DB_PATH)
            # Define the core fetching function for AWS, memoized by date
            # This function should ONLY be called for specific dates.
            # We add a dummy parameter `_` to allow easy cache busting for today
            @cache.memoize()
            def fetch_day_data_aws_cached(query_date_str, _=None):
                print(f"DB HIT (AWS): Fetching data for date {query_date_str}")
                try:
                    response = self.table.query(
                        KeyConditionExpression=Key('date').eq(query_date_str),
                        ExpressionAttributeNames={"#dt": "datetime"},
                        ProjectionExpression="#dt, Temperature, Humidity, eCO2"
                    )
                    items = response.get('Items', [])
                    if not items:
                         return pd.DataFrame(columns=['datetime', 'Temperature', 'Humidity', 'eCO2'])

                    df = pd.DataFrame(items)
                    # Convert numeric types here if necessary, before caching
                    for col in ['Temperature', 'Humidity', 'eCO2']:
                         df[col] = pd.to_numeric(df[col])
                    return df

                except Exception as e:
                    print(f"Error fetching data for {query_date_str} from AWS: {e}")
                    # Return empty DataFrame on error to avoid caching failures
                    return pd.DataFrame(columns=['datetime', 'Temperature', 'Humidity', 'eCO2'])

            self._fetch_day_data_aws_cached = fetch_day_data_aws_cached

        elif DB_TYPE == "LOCAL":
            self.conn = sqlite3.connect(DB_PATH, check_same_thread=False) 

            @cache.memoize()
            def fetch_day_data_local_cached(query_date_str, _=None):
                 print(f"DB HIT (Local): Fetching data for date {query_date_str}")
                 try:
                    query = f"SELECT Time as datetime, Temperature, Humidity, eCO2 FROM weather_data WHERE DATE(Time) = '{query_date_str}'"
                    df = pd.read_sql_query(query, self.conn)
                    if df.empty:
                         return pd.DataFrame(columns=['datetime', 'Temperature', 'Humidity', 'eCO2'])
                    
                    for col in ['Temperature', 'Humidity', 'eCO2']:
                         df[col] = pd.to_numeric(df[col], errors='coerce')
                    return df
                 except Exception as e:
                    print(f"Error fetching data for {query_date_str} from Local DB: {e}")
                    return pd.DataFrame(columns=['datetime', 'Temperature', 'Humidity', 'eCO2'])

            self._fetch_day_data_local_cached = fetch_day_data_local_cached
        else:
            raise ValueError("Unsupported DB_TYPE")

    def _get_fetch_function(self):
        """Returns the appropriate cached data fetching function based on DB_TYPE."""
        if DB_TYPE == "AWS":
            return self._fetch_day_data_aws_cached
        else:
            return self._fetch_day_data_local_cached

    def _fetch_data_for_day(self, query_date_str):
        """Fetch data for a single day. Handles cache clearing for the current day."""
        fetch_func = self._get_fetch_function()

        if query_date_str == datetime.now().strftime('%Y-%m-%d'):
            # Invalidate cache for today before fetching
            cache.delete_memoized(fetch_func, query_date_str)
            # Call with a changing dummy arg to ensure it's not memoized during this request if called multiple times today
            return fetch_func(query_date_str, _=datetime.now().timestamp())
        else:
            # For past dates, rely on the memoized function
            return fetch_func(query_date_str)

    def get_data(self, start_date_str, end_date_str):
        """Fetches data for a range of dates (inclusive). Leverages caching for individual past days."""
        all_data = []
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        current_date = start_date
        while current_date <= end_date:
            current_date_str = current_date.strftime('%Y-%m-%d')
            day_df = self._fetch_data_for_day(current_date_str)
            if not day_df.empty:
                all_data.append(day_df)
            current_date += timedelta(days=1)

        if not all_data:
            return pd.DataFrame()

        df = pd.concat(all_data, ignore_index=True)
        return df

    def get_all_historical_data(self):
        """Recursively fetches data day by day into the past until no more data is found. Leverages the daily cache."""
        all_data = []
        # Start from yesterday
        current_query_date = datetime.now().date() - timedelta(days=1)

        while True:
            query_date_str = current_query_date.strftime('%Y-%m-%d')
            print(f"Attempting to fetch historical data for: {query_date_str}")
            day_df = self._fetch_data_for_day(query_date_str) # This uses the cache

            if day_df.empty:
                print(f"No more historical data found before {query_date_str}.")
                break # Stop when no data is returned for a day

            all_data.append(day_df)

            # Move to the previous day
            current_query_date -= timedelta(days=1)

        if not all_data:
            return pd.DataFrame()

        df = pd.concat(all_data, ignore_index=True)
        return df

    def get_data_by_timerange(self, time_range):
        """Get data based on predefined ranges."""
        final_df = pd.DataFrame()
        now = datetime.now()
        today_str = now.strftime('%Y-%m-%d')

        if time_range == 0: # Current day
            print("Fetching data for: Today")
            final_df = self._fetch_data_for_day(today_str) # Fetches fresh data

        elif time_range == 1: # Current week
            print("Fetching data for: Current Week (inc. today)")
            start_date = now.date() - timedelta(days=7)
            start_date_str = start_date.strftime('%Y-%m-%d')
            final_df = self.get_data(start_date_str, today_str) # Fetches range, today will be fresh

        elif time_range > 1: # All time
            print("Fetching data for: All Time")
            # Get all historical data (uses cache, stops when no more data)
            historical_df = self.get_all_historical_data()
            # Get fresh data for today
            today_df = self._fetch_data_for_day(today_str)
            # Combine historical and today's data
            final_df = pd.concat([historical_df, today_df], ignore_index=True)

        # Final Processing (applied to all valid time_ranges)
        if not final_df.empty:

            final_df['Time'] = pd.to_datetime(final_df['datetime'])
            final_df.drop(columns=['datetime'], inplace=True, errors='ignore')

            # Convert numeric columns
            for col in ['Temperature', 'Humidity', 'eCO2']:
                if col in final_df.columns:
                     final_df[col] = pd.to_numeric(final_df[col])

            final_df.sort_values(by="Time", ascending=True, inplace=True)

            if time_range >= 1 and not final_df.empty:
                min_date_in_df = final_df["Time"].dt.date.min()
                final_df = final_df[final_df["Time"].dt.date != min_date_in_df]

        return final_df

    # Get the latest single reading
    def get_latest_values(self):
        """Gets the most recent data point for the current day."""
        if DB_TYPE == "LOCAL":
             # Optimized query for SQLite
             query = "SELECT * FROM weather_data ORDER BY Time DESC LIMIT 1"
             try:
                 df = pd.read_sql_query(query, self.conn)
                 if not df.empty:
                      df['Time'] = pd.to_datetime(df['Time'])
                      return df.iloc[0]
                 return None
             except Exception as e:
                 print(f"Error fetching latest value from Local DB: {e}")
                 return None

        today_df = self._fetch_data_for_day(datetime.now().strftime('%Y-%m-%d')) # Gets fresh data
        if not today_df.empty:
            today_df['Time'] = pd.to_datetime(today_df['datetime'])
            today_df.sort_values(by='Time', ascending=False, inplace=True)
            for col in ['Temperature', 'Humidity', 'eCO2']:
                if col in today_df.columns:
                     today_df[col] = pd.to_numeric(today_df[col], errors='coerce')
            return today_df.iloc[0]
        return None

    def close(self):
        if DB_TYPE == "LOCAL":
            self.conn.close()


pio.templates.default = "plotly_dark"

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
external_stylesheets = [dbc.themes.SOLAR, dbc_css]

chart_theme = {
    'margin': dict(l=0, r=0, t=0, b=0),
    'yaxis_title': None,
    'xaxis_title': None,
    'paper_bgcolor': "rgb(0, 0, 0, 0)",
    'plot_bgcolor': 'rgb(0, 0, 0, 0)',
    'legend': dict(bgcolor = 'rgb(0, 43, 54)'),
}

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                title="Weather Dashboard",
                meta_tags=[{"name": "viewport", "content": "margin=0"}],
                update_title=None,
               )

cache = Cache(app.server, config={'CACHE_TYPE': 'simple'})

temp_value = html.Div(id='temp_value', style={'font-size': '24px'})
humidity_value = html.Div(id='humidity_value', style={'font-size': '24px'})
eCO2_value = html.Div(id='eCO2_value', style={'font-size': '24px'})

app.layout = html.Div([
    html.H1('Weather Station Dashboard', className='text-center mb-4'),
    html.Div(id='browser-title', style={'display': 'none'}),
    html.Div(id='browser-title-values', style={'display': 'none'}),
    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.Div(temp_value, className='text-nowrap'),
                        html.Div(humidity_value, className='text-nowrap'),
                        html.Div(eCO2_value, className='text-nowrap')
                    ]),
                    className="mb-4"
                ),
                html.H5("Granularity"),
                dcc.Slider(
                    id='granularity-slider',
                    min=60,
                    max=3600,
                    step=None,
                    marks={
                        60: '1min',
                        600: '10min',
                        1800: '30min',
                        3600: '1h'
                    },
                    value=60
                ),
                html.H5("Time Range"),
                dcc.Slider(
                    id='current-time-range',
                    min=0,
                    max=2,
                    step=1,
                    marks={
                        0: 'Current Day',
                        1: 'Current Week',
                        2: 'All'
                    },
                    value=0,
                ),
                html.H5("Options"),
                dbc.Switch(
                    id='outdoor-toggle',
                    label='Show Outdoor Weather',
                    value=False,
                    inputClassName="mr-2"
                ),
                html.H5("Aggregation and Chart Type"),
                dbc.RadioItems(
                    id='aggregation-chart-selector',
                    options=[
                        {'label': 'Full Time Range Lines', 'value': 'full-line'},
                        {'label': '24h Overlay Lines', 'value': 'stacked-line'},
                        {'label': '24h Overlay Points', 'value': 'stacked-scatter'},
                        {'label': '24h Boxplot', 'value': 'median-box'}
                    ],
                    value='full-line'
                ),
            ], md=2, className='mb-3 mb-md-0'),
            dbc.Col([
                html.Div(
                    dcc.Graph(id='main-graph', config={'displayModeBar': False}, style={'height': '80vh'}),
                    id='chart-container'
                )
            ], md=9),
        ]),
    ], fluid=True, className='dbc'),
])

# update the browser title with the latest values
app.clientside_callback(
    """
    function(values) {
        const title = `${values} | Weather Dashboard`;
        document.title = title;
        return title;
    }
    """,
    Output("browser-title", "children"),
    [Input("browser-title-values", "children")]
)

# API calls for outdoor weather data
@cache.memoize(timeout=CACHE_TIMEOUT)
def get_outdoor_weather(time_range):
    '''Retrieves outdoor weather data and saves it to a parquet file for caching. The data is downloaded at most once per day.'''
    db = DatabaseHandler()
    min_date = db.get_data_by_timerange(time_range)["Time"].min().date()
    db.close()
    GATHER_DATA = True

    try:
        df_export = pd.read_parquet('weather_data_outdoor.parquet')
        most_recent_data = df_export['Time'].max().date()
        if most_recent_data == datetime.now().date():
            GATHER_DATA = False

    except FileNotFoundError:
        df_export = pd.DataFrame(columns=['Time', 'Temperature', 'Humidity', 'Precipitation'])


    if GATHER_DATA:
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=1, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        current_date = datetime.now().date()
        daily_data = pd.DataFrame()

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation"],
            "timezone": "auto",
            "start_date": min_date - timedelta(days=6),
            "end_date": current_date - timedelta(days=6),
        }

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        data_historical = response.Hourly()

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "minutely_15": ["temperature_2m", "relative_humidity_2m", "precipitation"],
            "timezone": "auto",
            "past_days": 5,
            "forecast_days": 1
        }

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        data_forecast = response.Minutely15()
        timezone_offset = pytz.timezone(response.Timezone()).utcoffset(datetime.now()).total_seconds()

        for data in [data_historical, data_forecast]:
            response_df = pd.DataFrame({
                "Time": pd.date_range(
                    start=pd.to_datetime(data.Time(), unit = "s") + pd.Timedelta(seconds=timezone_offset),
                    end=pd.to_datetime(data.TimeEnd(), unit = "s") + pd.Timedelta(seconds=timezone_offset),
                    freq=pd.Timedelta(seconds=data.Interval()),
                    inclusive="left"
                ),
                'Temperature': data.Variables(0).ValuesAsNumpy(),
                'Humidity': data.Variables(1).ValuesAsNumpy(),
                'Precipitation': data.Variables(2).ValuesAsNumpy()
            })
            daily_data = pd.concat([daily_data.dropna(), response_df.dropna()])

        df_export = pd.concat([df_export, daily_data]).drop_duplicates()
        df_export = df_export.sort_values(by='Time')
        df_export['Time'] = pd.to_datetime(df_export['Time'], unit='s', format='%Y-%m-%d %H:%M:%S')
        df_export.to_parquet('weather_data_outdoor.parquet')

    return df_export


# if the full time range is used, the granularity is reduced to avoid loading too many data points.
@app.callback(
    Output('granularity-slider', 'value'),
    [Input('current-time-range', 'value'),
    Input('granularity-slider', 'value')]
)   
def update_granularity_slider(current_time_range, current_granularity):
    if current_time_range == 2:
        if current_granularity < 1800:
            return 1800
        else:
            return current_granularity
    else:
        return current_granularity


# Update the values of the widgets
@app.callback(
    [Output('temp_value', 'children'),
     Output('humidity_value', 'children'),
     Output('eCO2_value', 'children'),
     Output('browser-title-values', 'children')],
    [Input('interval-component', 'n_intervals')]
)
@cache.memoize(timeout=CACHE_TIMEOUT)
def update_widget_values(n_intervals):
    db = DatabaseHandler()
    last_row = db.get_latest_values()
    db.close()
    
    if last_row["eCO2"] > 1000:
        title = f"+++AIR+++ {last_row['eCO2']:.0f}ppb | {last_row['Temperature']:.2f}°C | {last_row['Humidity']:.2f}%"
    else:
        title = f"{last_row['eCO2']:.0f}ppb | {last_row['Temperature']:.2f}°C | {last_row['Humidity']:.2f}%"

    return (f"Temperature: {last_row['Temperature']:.2f}°C",
            f"Humidity: {last_row['Humidity']:.2f}%",
            f"eCO2: {int(last_row['eCO2'])}ppb",
            title)


# Graph update function    
@app.callback(
    Output('main-graph', 'figure'),
    [Input('granularity-slider', 'value'),
     Input('current-time-range', 'value'),
     Input('aggregation-chart-selector', 'value'),
     Input('outdoor-toggle', 'value')]
)
@cache.memoize(timeout=CACHE_TIMEOUT)
def update_daily_graph(granularity, time_range, aggregation_chart_selector, include_outdoor):
    aggregation_type, chart_type = aggregation_chart_selector.split('-')
    db = DatabaseHandler()
    df = db.get_data_by_timerange(time_range)
    db.close()

    if include_outdoor:
        df_outdoor = get_outdoor_weather(time_range)
        #interpolate outdoor data according to granularity
        df_outdoor = df_outdoor.set_index('Time').resample(str(granularity) + "s").interpolate().reset_index()
        df = df.sort_values("Time")
        df = pd.merge_asof(df, df_outdoor, on="Time", suffixes=("", "_outdoor"), direction="nearest")

    df = df.set_index('Time').resample(str(granularity) + "s").median().reset_index()
    df["clock_time"] = df["Time"].dt.time

    unique_days = df["Time"].dt.date.unique()
    n_days = df["Time"].dt.dayofyear.nunique()

    temp_range = [df["Temperature"].min()*0.8, df["Temperature"].max()*1.1]
    hum_range = [df["Humidity"].min()*0.8, df["Humidity"].max()*1.1]
    eCO2_range = [df["eCO2"].min()*0.8, df["eCO2"].max()*1.1]

    if include_outdoor:
        temp_range = [min(temp_range[0], df["Temperature_outdoor"].min()*0.8), max(temp_range[1], df["Temperature_outdoor"].max()*1.1)]
        hum_range = [min(hum_range[0], df["Humidity_outdoor"].min()*0.8), max(hum_range[1], df["Humidity_outdoor"].max()*1.1)]


    # pre-define colors for the charts  
    color_dict = {
        "Temperature": {"color": (239, 85, 59), "range": temp_range},
        "Humidity": {"color": (99, 110, 250), "range": hum_range},
        "eCO2": {"color": (0, 204, 150), "range": eCO2_range},
        "Temperature_outdoor": {"color": (255, 165, 0), "range": temp_range},
        "Humidity_outdoor": {"color": (0, 190, 255), "range": hum_range}
    }

    # fade from white to the color_dict values by the number of days
    def fade_to_white(color, day_index, n_days):
        # Interpolate each RGB component towards 255
        return tuple(
            int(color_component + (255 - color_component) * (day_index / n_days))
            for color_component in color
        )

    colorscale_dict = {
        "Temperature": [f"rgb{fade_to_white(color_dict['Temperature']['color'], day_index, n_days)}" for day_index in range(n_days)],
        "Humidity": [f"rgb{fade_to_white(color_dict['Humidity']['color'], day_index, n_days)}" for day_index in range(n_days)],
        "eCO2": [f"rgb{fade_to_white(color_dict['eCO2']['color'], day_index, n_days)}" for day_index in range(n_days)],
        "Temperature_outdoor": [f"rgb{fade_to_white(color_dict['Temperature_outdoor']['color'], day_index, n_days)}" for day_index in range(n_days)],
        "Humidity_outdoor": [f"rgb{fade_to_white(color_dict['Humidity_outdoor']['color'], day_index, n_days)}" for day_index in range(n_days)]
    }
        
    figures = {}

    columns_to_plot = ["Temperature", "Humidity", "eCO2"]
    if include_outdoor:
        columns_to_plot.extend(["Temperature_outdoor", "Humidity_outdoor"])

    for column in columns_to_plot:
        figures[column] = go.Figure()
        if aggregation_type == "median":
            if chart_type == "box":
                figures[column].add_trace(go.Box(x=df["clock_time"], y=df[column], name=f"{column} (Sensor)" if "_outdoor" not in column else f"{column.replace('_outdoor', '')} (Outdoor)"))
        elif aggregation_type == "full":
            if chart_type == "line":
                figures[column].add_trace(go.Scatter(x=df["Time"], y=df[column], line_shape='spline', name=f"{column} (Sensor)" if "_outdoor" not in column else f"{column.replace('_outdoor', '')} (Outdoor)"))
        elif aggregation_type == "stacked":
            for day_index, day in enumerate(unique_days):
                df_day = df[df["Time"].dt.date == day]
                if not df_day[df_day["clock_time"].apply(lambda x: x.hour == 0)][column].isna().all():
                    trace = go.Scatter(
                        x=df_day["clock_time"],
                        y=df_day[column],
                        name=f"{str(day)} - {'Sensor' if '_outdoor' not in column else 'Outdoor'}",
                        line_shape='spline' if chart_type == 'line' else None,
                        mode="lines" if chart_type == 'line' else "markers",
                        line=dict(color=colorscale_dict[column][day_index]) if chart_type == 'line' else None,
                        marker=dict(color=colorscale_dict[column][day_index]) if chart_type == 'scatter' else None
                    )
                    figures[column].add_trace(trace)


    fig_merged = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, x_title="Time")

    for column_index, column_name in enumerate(["Temperature", "Humidity", "eCO2"]):
        if column_name in figures:
            for trace in figures[column_name].data:
                if aggregation_type != "stacked":
                    trace.line.color = f"rgb{color_dict[column_name]['color']}"
                    
                fig_merged.add_trace(trace, row=column_index+1, col=1)
            
            if include_outdoor and column_name != "eCO2":
                outdoor_column = f"{column_name}_outdoor"
                for trace in figures[outdoor_column].data:
                    if aggregation_type != "stacked":
                        trace.line.color = f"rgb{color_dict[outdoor_column]['color']}"

                    fig_merged.add_trace(trace, row=column_index+1, col=1)

            fig_merged.update_yaxes(title_text=column_name, row=column_index+1, col=1, range=color_dict[column_name]["range"])

    fig_merged.update_layout(**{**chart_theme, "showlegend": True, "yaxis_title": "Temperature"})

    if aggregation_type == "full":
        min_time = df["Time"].min()
        # end of most recent day
        max_time = df["Time"].max().replace(hour=23, minute=59, second=59)
        x0 = min_time
        x1 = max_time
    
    else:
        # use 0 and 24 hours as the limits for the x-axis
        x0 = 0
        x1 = 24*3600/granularity
        fig_merged.update_xaxes(dtick=x1/8, tickformat="%H:%M", row=3, col=1, range=[x0, x1])
        fig_merged.update_xaxes(matches='x')
        
    fig_merged.add_shape(type="rect", x0=x0, x1=x1, y0=20, y1=22, fillcolor="green", opacity=0.1, row=1, col=1)
    fig_merged.add_shape(type="rect", x0=x0, x1=x1, y0=40, y1=60, fillcolor="green", opacity=0.1, row=2, col=1)
    fig_merged.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=1000, fillcolor="green", opacity=0.1, row=3, col=1)

    return fig_merged

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8050, debug=True)
