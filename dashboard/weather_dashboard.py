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
    from boto3.dynamodb.conditions import Attr

########## Config values ##########
# Either "AWS" if you are using AWS DynamoDB or "LOCAL" for a SQLite database
DB_TYPE = os.getenv("DB_TYPE", "AWS")

# If your DB_TYPE is "AWS", enter the name of your table here
# If your DB_TYPE is "LOCAL", enter the path to your SQLite database here and make sure to include "weather_data.db" at the end
DB_PATH = os.getenv("DB_PATH", r"weather-data")

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
        else:
            self.conn = sqlite3.connect(DB_PATH)
            
    # Get the minimum date in the database
    def get_min_date(self):
        if DB_TYPE == "AWS":
            response = self.table.scan(
                FilterExpression=Attr("Time").gte("1970-01-01"),
                ExpressionAttributeNames={
                    "#ts": "Time"
                },
                ProjectionExpression="#ts"
            )
            min_date = min(item['Time'] for item in response['Items'])
            return datetime.strptime(min_date, '%Y-%m-%d %H:%M:%S').date()
        else:
            query = "SELECT MIN(Time) FROM weather_data"
            min_date = pd.read_sql_query(query, self.conn).iloc[0, 0]
            return datetime.strptime(min_date, '%Y-%m-%d %H:%M:%S').date()
            
    # Get the latest values from the database
    def get_latest_values(self):
        if DB_TYPE == "AWS":
            response = self.table.scan(
                FilterExpression=Attr("Time").gte("1970-01-01"),
                ExpressionAttributeNames={
                    "#ts": "Time"
                },
                ProjectionExpression="#ts, Temperature, Humidity, eCO2"
            )
            response = sorted(response['Items'], key=lambda x: x['Time'], reverse=True)
            return pd.DataFrame(response)
        else:
            query = "SELECT * FROM weather_data ORDER BY Time DESC LIMIT 1"
            return pd.read_sql_query(query, self.conn)
            
    def get_data_by_timerange(self, time_range):
        if DB_TYPE == "AWS":
            if time_range == 0: # Current day
                start_date = datetime.now().strftime('%Y-%m-%d')
            elif time_range == 1: # Current week
                start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            else: # All time
                start_date = "1970-01-01"
                
            response = self.table.scan(
                FilterExpression=Attr("Time").gte(start_date),
                ExpressionAttributeNames={
                    "#ts": "Time"
                },
                ProjectionExpression="#ts, Temperature, Humidity, eCO2"
            )
            df = pd.DataFrame(response['Items'])
            for col in ['Temperature', 'Humidity', 'eCO2']:
                df[col] = df[col].astype(float)
            df['Time'] = pd.to_datetime(df['Time'])
            
            if time_range == 1:
                df = df[df["Time"].dt.date != df["Time"].dt.date.min()]
                
            return df
        else:
            if time_range == 0:
                query = "SELECT * FROM weather_data WHERE DATE(Time) = DATE('now')"
            elif time_range == 1:
                query = "SELECT * FROM weather_data WHERE Time > DATE('now', '-7 days')"
            else:
                query = "SELECT * FROM weather_data"
                
            df = pd.read_sql_query(query, self.conn)
            df['Time'] = pd.to_datetime(df['Time'])
            
            if time_range != 0:
                df = df[df["Time"].dt.date != df["Time"].dt.date.min()]
                
            return df
            
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
def get_outdoor_weather():
    '''Retrieves outdoor weather data and saves it to a parquet file for caching. The data is downloaded at most once per day.'''
    db = DatabaseHandler()
    min_date = db.get_min_date()
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
    last_row = db.get_latest_values().iloc[0]
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
        df_outdoor = get_outdoor_weather()
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
    app.run_server(host="0.0.0.0", port=8050, debug=True)
