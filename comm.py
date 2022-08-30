import json
import logging

import requests

# Importing the API exception
from tb_rest_client.rest import ApiException
from tb_rest_client.rest_client_pe import *

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(module)s - %(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

data = [
    444,
    442,
    436,
    435,
    447,
    438,
    442,
    443,
    450,
    514,
    600,
    604,
    644,
    598,
    685,
    760,
    737,
    694,
    673,
    625,
    637,
    580,
    532,
    538,
    556,
    602,
    615,
    629,
    624,
    610,
    604,
    582,
    565,
    552,
    539,
    532,
    520,
    510,
    503,
    490,
    488,
    487,
    478,
    472,
    465,
    464,
    459,
    455,
    452,
    440,
    443,
    443,
    445,
    444,
    547,
    666,
    761,
    829,
    816,
    851,
    837,
    809,
    797,
    777,
    744,
    711,
    693,
    677,
    671,
    649,
    605,
    567,
    544,
    533,
    514,
    505,
    497,
    489,
    474,
    465,
    463,
    448,
    441,
    439,
    432,
    421,
    430,
    424,
    420,
    422,
    419,
    412,
    417,
    415,
    414,
    412,
    415,
    416,
    493,
    567,
    690,
    787,
    865,
    557,
    582,
    528,
    450,
    508,
    526,
    531,
    609,
    711,
    491,
    617,
    623,
    552,
    512,
    481,
    465,
    448,
    446,
    435,
    422,
    423,
    421,
    415,
    416,
    410,
    406,
    404,
    400,
    400,
    400,
    401,
    402,
    399,
    404,
    410,
    406,
    419,
    434,
    424,
    433,
    429,
]

# ThingsBoard REST API URL
url = "https://thingsboard.cloud"
# Default Tenant Administrator credentials
username = "pfernandez@mat.upv.es"
password = "123456"

uuid = "dc8b47a0-186d-11ec-a9e6-556e8dbef35c"
# Creating the REST client object with context manager to get auto token refresh
while True:
    with RestClientPE(base_url=url) as rest_client:
        try:
            # Auth with credentials
            rest_client.login(username=username, password=password)

            # find device by device id
            device = rest_client.get_device_by_id(uuid)

            # Get device shared attributes
            latest_value = (
                rest_client.telemetry_controller.get_latest_timeseries_using_get(
                    "DEVICE", "dc8b47a0-186d-11ec-a9e6-556e8dbef35c", keys="CO2"
                )
            )
            latest_value = latest_value["CO2"]

        except ApiException as e:
            logging.exception(e)
    data.append(int(latest_value[0]["value"]))
    data.pop(0)
    logging.info("Data updated: \n%r", data)

    # Comm with Flask Server
    json = {"data": data}
    flask_server = "https://flaskai-app.heroku.com/predict"
    response = requests.post(flask_server, json=json)

    logging.info("SENT")

    sleep(1800)  # 1800
