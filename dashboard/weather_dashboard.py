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
import threading
import time
import queue
from dash import dcc, html, clientside_callback
from dash.dependencies import Output, Input, State
from plotly.subplots import make_subplots
from flask_caching import Cache
from retry_requests import retry
from datetime import datetime, timedelta
from boto3.dynamodb.conditions import Key
from dotenv import load_dotenv

load_dotenv()

########## Config values ##########
# Either "AWS" if you are using AWS DynamoDB or "LOCAL" for a SQLite database
DB_TYPE = os.getenv("DB_TYPE", "AWS")

# If your DB_TYPE is "AWS", enter the name of your table here
# If your DB_TYPE is "LOCAL", enter the path to your SQLite database here and make sure to include "sensor_data.db" at the end
DB_PATH = os.getenv("DB_PATH", r"sensor-data")
REGION_NAME = os.getenv("AWS_REGION", "us-east-1")

# Coordinates for API weather data
LATITUDE = float(os.getenv("LATITUDE", 49.0094))
LONGITUDE = float(os.getenv("LONGITUDE", 8.4044))

# Create a global queue for new data from the stream
new_data_queue = queue.Queue()

notification_cooldown = timedelta(minutes=15)


class DatabaseHandler:
    def __init__(self):
        self.today_df = pd.DataFrame(
            columns=["Time", "Temperature", "Humidity", "eCO2"]
        )
        self.today_date_str = datetime.now().strftime("%Y-%m-%d")

        if DB_TYPE == "AWS":
            self.dynamodb = boto3.resource("dynamodb", region_name=REGION_NAME)
            self.table = self.dynamodb.Table(DB_PATH)

            @cache.memoize()
            def fetch_day_data_aws_cached(query_date_str, _=None):
                try:
                    response = self.table.query(
                        KeyConditionExpression=Key("date").eq(query_date_str),
                        ExpressionAttributeNames={"#dt": "datetime"},
                        ProjectionExpression="#dt, Temperature, Humidity, eCO2",
                    )
                    items = response.get("Items", [])
                    df = normalize_aws_df(pd.DataFrame(items))

                    if query_date_str == self.today_date_str:
                        self.today_df = df.copy()

                    return df
                except Exception as e:
                    print(f"Error fetching data for {query_date_str} from AWS: {e}")
                    return pd.DataFrame(
                        columns=["Time", "Temperature", "Humidity", "eCO2"]
                    )

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
                        return pd.DataFrame(
                            columns=["Time", "Temperature", "Humidity", "eCO2"]
                        )

                    for col in ["Temperature", "Humidity", "eCO2"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                    if query_date_str == self.today_date_str:
                        self.today_df = df.copy()
                    return df
                except Exception as e:
                    print(
                        f"Error fetching data for {query_date_str} from Local DB: {e}"
                    )
                    return pd.DataFrame(
                        columns=["Time", "Temperature", "Humidity", "eCO2"]
                    )

            self._fetch_day_data_local_cached = fetch_day_data_local_cached
        else:
            raise ValueError("Unsupported DB_TYPE")

        # Initialize data on startup
        self._fetch_data_for_day(self.today_date_str)

    def _get_fetch_function(self):
        """Returns the relevant cached data fetching function based on DB_TYPE."""
        if DB_TYPE == "AWS":
            return self._fetch_day_data_aws_cached
        else:
            return self._fetch_day_data_local_cached

    def _check_date_rollover(self):
        """Checks if the date has changed and resets today_df if needed."""
        current_date_str = datetime.now().strftime("%Y-%m-%d")
        if current_date_str != self.today_date_str:
            print(f"Date rolled over from {self.today_date_str} to {current_date_str}")
            self.today_date_str = current_date_str
            self.today_df = pd.DataFrame(
                columns=["Time", "Temperature", "Humidity", "eCO2"]
            )
            self._fetch_data_for_day(self.today_date_str)

    def _fetch_data_for_day(self, query_date_str):
        """Fetch data for a single day. Handles cache clearing for the current day."""
        fetch_func = self._get_fetch_function()
        self._check_date_rollover()

        if query_date_str == self.today_date_str:
            use_cached_today = False
            if not self.today_df.empty and "Time" in self.today_df.columns:
                try:
                    latest_time = pd.to_datetime(self.today_df["Time"]).max()
                    time_diff = datetime.now() - latest_time
                    if time_diff.total_seconds() / 60 < 3:
                        # print("Using in-memory today_df - last update less than 3 minutes ago")
                        use_cached_today = True
                except Exception as e:
                    print(f"Error comparing times for today's cache: {e}")

            if use_cached_today:
                return self.today_df.copy()

            else:
                print(
                    "NOT USING CACHE, fetching fresh data for today: "
                    + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                # Invalidate cache before fetching
                cache.delete_memoized(fetch_func, query_date_str)
                return fetch_func(query_date_str, _=datetime.now().timestamp())

        else:
            # For past dates, rely on the memoized function
            return fetch_func(query_date_str)

    def update_today_df_from_stream(self, new_record_dict):
        """Appends a new record from the stream to the internal today_df."""
        self._check_date_rollover()

        if new_record_dict.get("date") == self.today_date_str:
            new_row_df = normalize_aws_df(pd.DataFrame([new_record_dict]))
        if not new_row_df.empty:
            self.today_df = pd.concat([self.today_df, new_row_df], ignore_index=True)
            self.today_df.sort_values(by="Time", inplace=True)
            self.today_df.drop_duplicates(subset=["Time"], keep="last", inplace=True)

    def get_latest_values(self):
        """Gets the most recent data point for the current day from internal df or DB."""
        self._check_date_rollover()

        if DB_TYPE == "LOCAL":
            query = "SELECT * FROM weather_data ORDER BY Time DESC LIMIT 1"
            try:
                df = pd.read_sql_query(query, self.conn)
                if not df.empty:
                    df["Time"] = pd.to_datetime(df["Time"])
                    return df.iloc[0]
                return None
            except Exception as e:
                print(f"Error fetching latest value from Local DB: {e}")
                return None

        # For AWS, rely on the internal today_df first
        if not self.today_df.empty:
            return self.today_df.iloc[-1].copy()

        # If today_df is empty, fetch from DB
        else:
            latest_from_db = self._fetch_data_for_day(self.today_date_str)
            if not latest_from_db.empty:
                return latest_from_db.iloc[-1].copy()
            return None

    def get_data(self, start_date_str, end_date_str):
        """Fetches data for a range of dates (inclusive). Leverages caching for individual past days."""
        all_data = []
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

        current_date = start_date
        while current_date <= end_date:
            current_date_str = current_date.strftime("%Y-%m-%d")
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
            query_date_str = current_query_date.strftime("%Y-%m-%d")
            print(f"Attempting to fetch historical data for: {query_date_str}")
            day_df = self._fetch_data_for_day(query_date_str)  # Use the cache

            if day_df.empty:
                print(f"No more historical data found before {query_date_str}.")
                break

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
        today_str = now.strftime("%Y-%m-%d")

        if time_range == 0:  # Current day
            final_df = self._fetch_data_for_day(today_str)

        elif time_range == 1:  # Current week
            start_date = now.date() - timedelta(days=7)
            start_date_str = start_date.strftime("%Y-%m-%d")
            final_df = self.get_data(start_date_str, today_str)

        elif time_range > 1:  # All time
            historical_df = self.get_all_historical_data()
            self.today_df = self._fetch_data_for_day(today_str)
            final_df = pd.concat([historical_df, self.today_df], ignore_index=True)

        if not final_df.empty:
            final_df = normalize_aws_df(final_df)

            if time_range >= 1 and not final_df.empty:
                min_date_in_df = final_df["Time"].dt.date.min()
                final_df = final_df[final_df["Time"].dt.date != min_date_in_df]

        return final_df

    def close(self):
        if DB_TYPE == "LOCAL":
            self.conn.close()


def normalize_aws_df(aws_df):
    try:
        aws_df["Time"] = pd.to_datetime(aws_df["datetime"])
    except KeyError:
        aws_df["Time"] = pd.to_datetime(aws_df["Time"])

    aws_df.drop(columns=["datetime", "date", "TVOC"], inplace=True, errors="ignore")

    for col in ["Temperature", "Humidity", "eCO2"]:
        if col in aws_df.columns:
            aws_df[col] = pd.to_numeric(aws_df[col])

    aws_df.sort_values(by="Time", inplace=True)
    return aws_df


def get_latest_iterator():
    dynamodb = boto3.client("dynamodb", region_name=REGION_NAME)
    streams = boto3.client("dynamodbstreams", region_name=REGION_NAME)
    stream_arn = dynamodb.describe_table(TableName=DB_PATH)["Table"]["LatestStreamArn"]
    stream_desc = streams.describe_stream(StreamArn=stream_arn)
    shards = stream_desc["StreamDescription"]["Shards"]

    if not shards:
        print("No shards found.")
        return None

    latest_shard = shards[-1]["ShardId"]

    iterator = streams.get_shard_iterator(
        StreamArn=stream_arn, ShardId=latest_shard, ShardIteratorType="LATEST"
    )["ShardIterator"]

    return iterator


new_data_queue = queue.Queue()


def listen_to_stream(db_handler, data_queue):
    streams_client = boto3.client("dynamodbstreams", region_name=REGION_NAME)
    shard_iterator = None

    while True:
        try:
            if not shard_iterator:
                shard_iterator = get_latest_iterator()

            out = streams_client.get_records(ShardIterator=shard_iterator, Limit=10)
            records = out.get("Records", [])

            for record in records:
                if record["eventName"] == "INSERT":
                    new_image = record["dynamodb"]["NewImage"]
                    parsed = {k: list(v.values())[0] for k, v in new_image.items()}

                    if parsed.get("date") == db_handler.today_date_str:
                        # print(f"New record: {parsed}")
                        data_queue.put(parsed)
                        db_handler.update_today_df_from_stream(parsed)

            shard_iterator = out["NextShardIterator"]
            time.sleep(5)

        except Exception as e:
            print(f"Error getting shard iterator: {e} \nUsing new iterator")
            time.sleep(5)
            shard_iterator = None


def start_stream_listener(db_handler, data_queue):
    global stream_thread
    if stream_thread is None or not stream_thread.is_alive():
        stream_thread = threading.Thread(
            target=listen_to_stream, args=(db_handler, data_queue), daemon=True
        )
        stream_thread.start()
        print("Stream listener thread started")
    else:
        print("Stream listener already running.")


pio.templates.default = "plotly_dark"

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
external_stylesheets = [dbc.themes.SOLAR, dbc_css]

chart_theme = {
    "margin": dict(l=0, r=0, t=0, b=0),
    "yaxis_title": None,
    "xaxis_title": None,
    "paper_bgcolor": "rgb(0, 0, 0, 0)",
    "plot_bgcolor": "rgb(0, 0, 0, 0)",
    "legend": dict(bgcolor="rgb(0, 43, 54)"),
}

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    title="Weather Dashboard",
    meta_tags=[{"name": "viewport", "content": "margin=0"}],
    update_title=None,
)

cache = Cache(app.server, config={"CACHE_TYPE": "simple"})

db = DatabaseHandler()

if DB_TYPE == "AWS":
    stream_thread = None
    start_stream_listener(db, new_data_queue)

temp_value = html.Div(id="temp_value", style={"font-size": "24px"})
humidity_value = html.Div(id="humidity_value", style={"font-size": "24px"})
eCO2_value = html.Div(id="eCO2_value", style={"font-size": "24px"})

app.layout = html.Div(
    [
        html.H1("Weather Station Dashboard", className="text-center mb-4"),
        html.Div(id="browser-title", style={"display": "none"}),
        html.Div(id="browser-title-values", style={"display": "none"}),
        html.Div(id="current-data-store", style={"display": "none"}),
        html.Div(id="notification-output"),
        dcc.Store(id="time-range-store", data={"time-range": None}),
        dcc.Store(id="aq-notification-trigger"),
        dcc.Store(id="notification-permission"),
        dcc.Store(
            id="last-notification-store", storage_type="memory", data=datetime.min
        ),
        dcc.Interval(id="update-interval", interval=10000, n_intervals=0),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                temp_value, className="text-nowrap"
                                            ),
                                            html.Div(
                                                humidity_value, className="text-nowrap"
                                            ),
                                            html.Div(
                                                eCO2_value, className="text-nowrap"
                                            ),
                                        ]
                                    ),
                                    className="mb-4",
                                ),
                                html.H5("Granularity"),
                                dcc.Slider(
                                    id="granularity-slider",
                                    min=60,
                                    max=3600,
                                    step=None,
                                    marks={
                                        60: "1min",
                                        600: "10min",
                                        1800: "30min",
                                        3600: "1h",
                                    },
                                    value=60,
                                ),
                                html.H5("Time Range"),
                                dcc.Slider(
                                    id="current-time-range",
                                    min=0,
                                    max=2,
                                    step=1,
                                    marks={
                                        0: "Current Day",
                                        1: "Current Week",
                                        2: "All",
                                    },
                                    value=0,
                                ),
                                html.H5("Options"),
                                dbc.Switch(
                                    id="outdoor-toggle",
                                    label="Show Outdoor Weather",
                                    value=False,
                                    inputClassName="mr-2",
                                ),
                                dbc.Switch(
                                    id="notify-toggle",
                                    label="Air Quality Notifications",
                                    value=False,
                                    inputClassName="mr-2",
                                ),
                                html.H5("Aggregation and Chart Type"),
                                dbc.RadioItems(
                                    id="aggregation-chart-selector",
                                    options=[
                                        {
                                            "label": "Full Time Range Lines",
                                            "value": "full-line",
                                        },
                                        {
                                            "label": "24h Overlay Lines",
                                            "value": "stacked-line",
                                        },
                                        {
                                            "label": "24h Overlay Points",
                                            "value": "stacked-scatter",
                                        },
                                        {"label": "24h Boxplot", "value": "median-box"},
                                    ],
                                    value="full-line",
                                ),
                            ],
                            md=2,
                            className="mb-3 mb-md-0",
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    dcc.Graph(
                                        id="main-graph",
                                        config={"displayModeBar": False},
                                        style={"height": "80vh"},
                                    ),
                                    id="chart-container",
                                )
                            ],
                            md=9,
                        ),
                    ]
                ),
            ],
            fluid=True,
            className="dbc",
        ),
    ]
)

# Update the browser title with the latest values
clientside_callback(
    """
    function(values) {
        const title = `${values} | Weather Dashboard`;
        document.title = title;
        return title;
    }
    """,
    Output("browser-title", "children"),
    [Input("browser-title-values", "children")],
)

# Request notification permission when the toggle is switched on
clientside_callback(
    """
    async function(toggleValue) {
        await Notification.requestPermission();

        return null;
    }
    """,
    Output("notification-permission", "data"),
    Input("notify-toggle", "value"),
    prevent_initial_call=True,
)

# Trigger the notification when the eCO2 level is high
clientside_callback(
    """
    function(message, notifyToggle) {
        if (message) {
            if (Notification.permission === 'granted' && notifyToggle) {
                new Notification("Air Quality Alert", { body: message });
            }
        }
        return null;
    }
    """,
    Output("notification-output", "children"),
    Input("aq-notification-trigger", "data"),
    Input("notify-toggle", "value"),
    prevent_initial_call=True,
)


@cache.memoize()
def get_outdoor_weather(time_range, _=None):
    """Retrieves and caches outdoor weather data in memory using Flask-Caching."""
    min_date = db.get_data_by_timerange(time_range)["Time"].min().date()
    current_date = datetime.now().date()
    now = datetime.now()

    # Set up cache and retry
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=1, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    def fetch_today_data():
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "minutely_15": ["temperature_2m", "relative_humidity_2m"],
                "timezone": "auto",
                "past_days": 1,
                "forecast_days": 1,
            }

            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            data_forecast = response.Minutely15()
            timezone_offset = (
                pytz.timezone(response.Timezone()).utcoffset(now).total_seconds()
            )

            return pd.DataFrame(
                {
                    "Time": pd.date_range(
                        start=pd.to_datetime(data_forecast.Time(), unit="s")
                        + pd.Timedelta(seconds=timezone_offset),
                        end=pd.to_datetime(data_forecast.TimeEnd(), unit="s")
                        + pd.Timedelta(seconds=timezone_offset),
                        freq=pd.Timedelta(seconds=data_forecast.Interval()),
                        inclusive="left",
                    ),
                    "Temperature": data_forecast.Variables(0).ValuesAsNumpy(),
                    "Humidity": data_forecast.Variables(1).ValuesAsNumpy(),
                    # 'Precipitation': data_forecast.Variables(2).ValuesAsNumpy()
                }
            ).dropna()
        except Exception as e:
            print(f"Error fetching today's weather data: {e}")
            return pd.DataFrame(columns=["Time", "Temperature", "Humidity"])

    # Load full dataset from cache
    outdoor_data = pd.DataFrame(columns=["Time", "Temperature", "Humidity"])

    try:
        print("Historical weather data refetched")
        # Historical data
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": ["temperature_2m", "relative_humidity_2m"],
            "timezone": "auto",
            "start_date": min_date - timedelta(days=6),
            "end_date": current_date - timedelta(days=2),
        }

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        data_historical = response.Hourly()
        timezone_offset = (
            pytz.timezone(response.Timezone()).utcoffset(now).total_seconds()
        )

        historical_data = pd.DataFrame(
            {
                "Time": pd.date_range(
                    start=pd.to_datetime(data_historical.Time(), unit="s")
                    + pd.Timedelta(seconds=timezone_offset),
                    end=pd.to_datetime(data_historical.TimeEnd(), unit="s")
                    + pd.Timedelta(seconds=timezone_offset),
                    freq=pd.Timedelta(seconds=data_historical.Interval()),
                    inclusive="left",
                ),
                "Temperature": data_historical.Variables(0).ValuesAsNumpy(),
                "Humidity": data_historical.Variables(1).ValuesAsNumpy(),
                # 'Precipitation': data_historical.Variables(2).ValuesAsNumpy()
            }
        ).dropna()

        # Load today's data from forecast
        today_data = fetch_today_data()

        full_data = pd.concat([historical_data, today_data])

        full_data["Time"] = pd.to_datetime(full_data["Time"])
        outdoor_data = full_data.sort_values(by="Time")

    except Exception as e:
        print(f"Error fetching outdoor weather data: {e}")
        return pd.DataFrame(columns=["Time", "Temperature", "Humidity"])

    return outdoor_data.drop_duplicates()


# If the full time range is used, the granularity is reduced to avoid loading too many data points.
@app.callback(
    Output("granularity-slider", "value"),
    [Input("current-time-range", "value"), Input("granularity-slider", "value")],
)
def update_granularity_slider(current_time_range, current_granularity):
    if current_time_range == 2:
        if current_granularity < 1800:
            return 1800
        else:
            return current_granularity
    else:
        return current_granularity


@app.callback(
    [
        Output("temp_value", "children"),
        Output("humidity_value", "children"),
        Output("eCO2_value", "children"),
        Output("browser-title-values", "children"),
        Output("aq-notification-trigger", "data"),
        Output("last-notification-store", "data"),
    ],
    [Input("update-interval", "n_intervals")],
    [State("last-notification-store", "data")],
)
def update_widget_values(n_intervals, last_aq_notification):
    last_row = db.get_latest_values()

    if last_row is None:
        return (
            "Temperature: --°C",
            "Humidity: --%",
            "eCO2: -- ppb",
            "--°C | --% | -- ppb",
            None,
        )

    temperature = f"{last_row['Temperature']:.2f}°C"
    humidity = f"{last_row['Humidity']:.2f}%"
    eco2 = int(last_row["eCO2"])
    title = f"{eco2}ppb | {temperature} | {humidity}"

    notify = None
    now = datetime.now()
    last_aq_notification = datetime.fromisoformat(last_aq_notification)
    if eco2 > 1000 and (now - last_aq_notification) > notification_cooldown:
        last_aq_notification = now
        title = f"+++AIR+++ {title}"
        notify = f"⚠️ High eCO2 levels: {eco2}ppb"

    return (
        f"Temperature: {temperature}",
        f"Humidity: {humidity}",
        f"eCO2: {eco2}ppb",
        title,
        notify,
        last_aq_notification,
    )


# Process new data from stream for the graph updates
@app.callback(
    Output("current-data-store", "children"), [Input("update-interval", "n_intervals")]
)
def process_new_data_for_graphs(n_intervals):
    try:
        updated = False
        while not new_data_queue.empty():
            new_record = new_data_queue.get_nowait()
            # print(f"Processing new record from queue for graphs: {new_record}")
            updated = True

        # Return a timestamp to trigger the graph update callback if there are updates
        if updated:
            return str(datetime.now().timestamp())
        return "no-update"
    except queue.Empty:
        return "no-update"


@app.callback(
    Output("main-graph", "figure"),
    Output("time-range-store", "data"),
    Input("granularity-slider", "value"),
    Input("current-time-range", "value"),
    Input("aggregation-chart-selector", "value"),
    Input("outdoor-toggle", "value"),
    Input("current-data-store", "children"),
    State("main-graph", "relayoutData"),
    State("time-range-store", "data"),
)
def update_daily_graph(
    granularity,
    time_range,
    aggregation_chart_selector,
    include_outdoor,
    data_store_trigger,
    relayout_data,
    previous_time_range,
):
    aggregation_type, chart_type = aggregation_chart_selector.split("-")
    df = db.get_data_by_timerange(time_range)

    if include_outdoor:
        df_outdoor = get_outdoor_weather(time_range, datetime.now().date())
        # interpolate outdoor data according to granularity
        df_outdoor = (
            df_outdoor.set_index("Time")
            .resample(str(granularity) + "s")
            .interpolate()
            .reset_index()
        )
        df = df.sort_values("Time")
        df = pd.merge_asof(
            df, df_outdoor, on="Time", suffixes=("", "_outdoor"), direction="nearest"
        )

    df = df.set_index("Time").resample(str(granularity) + "s").median().reset_index()
    df["clock_time"] = df["Time"].dt.time

    unique_days = df["Time"].dt.date.unique()
    n_days = df["Time"].dt.dayofyear.nunique()

    temp_range = [df["Temperature"].min() * 0.8, df["Temperature"].max() * 1.1]
    hum_range = [df["Humidity"].min() * 0.8, df["Humidity"].max() * 1.1]
    eCO2_range = [df["eCO2"].min() * 0.8, df["eCO2"].max() * 1.1]

    if include_outdoor:
        temp_range = [
            min(temp_range[0], df["Temperature_outdoor"].min() * 0.8),
            max(temp_range[1], df["Temperature_outdoor"].max() * 1.1),
        ]
        hum_range = [
            min(hum_range[0], df["Humidity_outdoor"].min() * 0.8),
            max(hum_range[1], df["Humidity_outdoor"].max() * 1.1),
        ]

    # pre-define colors for the charts
    color_dict = {
        "Temperature": {"color": (239, 85, 59), "range": temp_range},
        "Humidity": {"color": (99, 110, 250), "range": hum_range},
        "eCO2": {"color": (0, 204, 150), "range": eCO2_range},
        "Temperature_outdoor": {"color": (255, 165, 0), "range": temp_range},
        "Humidity_outdoor": {"color": (0, 190, 255), "range": hum_range},
    }

    # fade from white to the color_dict values by the number of days
    def fade_to_white(color, day_index, n_days):
        # Interpolate each RGB component towards 255
        return tuple(
            int(color_component + (255 - color_component) * (day_index / n_days))
            for color_component in color
        )

    colorscale_dict = {
        "Temperature": [
            f"rgb{fade_to_white(color_dict['Temperature']['color'], day_index, n_days)}"
            for day_index in range(n_days)
        ],
        "Humidity": [
            f"rgb{fade_to_white(color_dict['Humidity']['color'], day_index, n_days)}"
            for day_index in range(n_days)
        ],
        "eCO2": [
            f"rgb{fade_to_white(color_dict['eCO2']['color'], day_index, n_days)}"
            for day_index in range(n_days)
        ],
        "Temperature_outdoor": [
            f"rgb{fade_to_white(color_dict['Temperature_outdoor']['color'], day_index, n_days)}"
            for day_index in range(n_days)
        ],
        "Humidity_outdoor": [
            f"rgb{fade_to_white(color_dict['Humidity_outdoor']['color'], day_index, n_days)}"
            for day_index in range(n_days)
        ],
    }

    figures = {}

    columns_to_plot = ["Temperature", "Humidity", "eCO2"]
    if include_outdoor:
        columns_to_plot.extend(["Temperature_outdoor", "Humidity_outdoor"])

    for column in columns_to_plot:
        figures[column] = go.Figure()
        if aggregation_type == "median":
            if chart_type == "box":
                figures[column].add_trace(
                    go.Box(
                        x=df["clock_time"],
                        y=df[column],
                        name=f"{column} (Sensor)"
                        if "_outdoor" not in column
                        else f"{column.replace('_outdoor', '')} (Outdoor)",
                    )
                )
        elif aggregation_type == "full":
            if chart_type == "line":
                figures[column].add_trace(
                    go.Scatter(
                        x=df["Time"],
                        y=df[column],
                        line_shape="spline",
                        name=f"{column} (Sensor)"
                        if "_outdoor" not in column
                        else f"{column.replace('_outdoor', '')} (Outdoor)",
                    )
                )
        elif aggregation_type == "stacked":
            for day_index, day in enumerate(unique_days):
                df_day = df[df["Time"].dt.date == day]
                if (
                    not df_day[df_day["clock_time"].apply(lambda x: x.hour == 0)][
                        column
                    ]
                    .isna()
                    .all()
                ):
                    trace = go.Scatter(
                        x=df_day["clock_time"],
                        y=df_day[column],
                        name=f"{str(day)} - {'Sensor' if '_outdoor' not in column else 'Outdoor'}",
                        line_shape="spline" if chart_type == "line" else None,
                        mode="lines" if chart_type == "line" else "markers",
                        line=dict(color=colorscale_dict[column][day_index])
                        if chart_type == "line"
                        else None,
                        marker=dict(color=colorscale_dict[column][day_index])
                        if chart_type == "scatter"
                        else None,
                    )
                    figures[column].add_trace(trace)

    fig_merged = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, x_title="Time"
    )

    for column_index, column_name in enumerate(["Temperature", "Humidity", "eCO2"]):
        if column_name in figures:
            for trace in figures[column_name].data:
                if aggregation_type != "stacked":
                    trace.line.color = f"rgb{color_dict[column_name]['color']}"

                fig_merged.add_trace(trace, row=column_index + 1, col=1)

            if include_outdoor and column_name != "eCO2":
                outdoor_column = f"{column_name}_outdoor"
                for trace in figures[outdoor_column].data:
                    if aggregation_type != "stacked":
                        trace.line.color = f"rgb{color_dict[outdoor_column]['color']}"

                    fig_merged.add_trace(trace, row=column_index + 1, col=1)

            fig_merged.update_yaxes(
                title_text=column_name,
                row=column_index + 1,
                col=1,
                range=color_dict[column_name]["range"],
            )

    fig_merged.update_layout(
        **{**chart_theme, "showlegend": True, "yaxis_title": "Temperature"}
    )

    if aggregation_type == "full":
        min_time = df["Time"].min()
        # end of most recent day
        max_time = df["Time"].max().replace(hour=23, minute=59, second=59)
        x0 = min_time
        x1 = max_time

    else:
        # use 0 and 24 hours as the limits for the x-axis
        x0 = 0
        x1 = 24 * 3600 / granularity
        fig_merged.update_xaxes(
            dtick=x1 / 8, tickformat="%H:%M", row=3, col=1, range=[x0, x1]
        )
        fig_merged.update_xaxes(matches="x")

    fig_merged.add_shape(
        type="rect",
        x0=x0,
        x1=x1,
        y0=20,
        y1=22,
        fillcolor="green",
        opacity=0.1,
        row=1,
        col=1,
    )
    fig_merged.add_shape(
        type="rect",
        x0=x0,
        x1=x1,
        y0=40,
        y1=60,
        fillcolor="green",
        opacity=0.1,
        row=2,
        col=1,
    )
    fig_merged.add_shape(
        type="rect",
        x0=x0,
        x1=x1,
        y0=0,
        y1=1000,
        fillcolor="green",
        opacity=0.1,
        row=3,
        col=1,
    )

    previous_time_range = dash.callback_context.states["time-range-store.data"][
        "time-range"
    ]

    if relayout_data and time_range == previous_time_range:
        layout_updates = {}

        for key, value in relayout_data.items():
            if ".range[" in key:
                axis_key, range_idx = key.split(".range[")
                range_idx = int(range_idx.rstrip("]"))

                if axis_key not in layout_updates:
                    layout_updates[axis_key] = [None, None]

                layout_updates[axis_key][range_idx] = value

        for axis, range_vals in layout_updates.items():
            fig_merged.update_layout({axis: dict(range=range_vals)})

    return fig_merged, {"time-range": time_range}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
