from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_time_interval
from datetime import timedelta
import logging
from .const import DOMAIN, MAX_FRAME_NUM
from .prediction import update_prediction_with_detections, is_failing, VISUALIZATION_THRESH

LOGGER = logging.getLogger(__package__)

PLATFORMS = [Platform.NUMBER, Platform.CAMERA, Platform.SWITCH, Platform.SENSOR]

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the Spaghetti Detection integration."""
    camera_entity_id = entry.data["camera_entity"]  # Required: Get the camera entity ID from the config entry data
    update_interval = entry.data.get("update_interval", 60) # Optional: Time in seconds between updates
    obico_ml_api_host = entry.data.get("obico_ml_api_host", "http://127.0.0.1:3333") # Optional: Obico ML API host
    obico_ml_api_token = entry.data.get("obico_ml_api_token", "obico_api_secret") # Optional: Obico ML API token
    device_name = entry.data["device_name"] # Required: The name of the printer device to monitor and interact with

    # Initialize the domain data dictionary
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}

    # Set the device name in hass.data[DOMAIN]
    hass.data[DOMAIN]["device_name"] = device_name

    # Initialize variables
    ewm_mean = 0
    rolling_mean_short = 0
    rolling_mean_long = 0
    current_frame_number = 0
    lifetime_frame_number = 0


    ################ from upload_print() from obico-server/backend/app/views/web_views.py
    ################ and preprocess_timelapse() from obico-server/backend/app/tasks.py
    ################ and detect_timelapse() from obico-server/backend/app/tasks.py
    async def processImage(request):
        if not hass.data[DOMAIN]["active"]:
            return

        # Get the image from the camera entity
        camera = hass.data[DOMAIN]["camera"]
        try:
            image = await camera.async_camera_image()
        except Exception as err:
            LOGGER.error("Failed to get image from camera: %s", err)
            return

        predictions = []
        last_prediction = PrinterPrediction()  # TO DO - need to implement - see obico-server/backend/app/models/other_models.py

        # Make the API call to the Obico ML server
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{obico_ml_api_host}/check/", params={'entity_id': camera_entity_id}, headers={"Authorization": f"Bearer {obico_ml_api_token}"}) as response:
                    response.raise_for_status()
                    result = await response.json()
                    detections = result['detections']
            except aiohttp.ClientError as e:
                LOGGER.error("Failed to make API call to Obico ML server: %s", e)
                return
            except Exception as e:
                LOGGER.error("Unexpected error: %s", e)
                return
        update_prediction_with_detections(last_prediction, detections)
        predictions.append(last_prediction)
        if is_failing(last_prediction, 1, escalating_factor=1):
            # *** TO DO *** Update the Printer Prediction
        last_prediction = copy.deepcopy(last_prediction)
        # *** TO DO *** Update the Camera Image
        predictions_json = serializers.serialize("json", predictions)


















    async def spaghetti_detection_handler(now):
        nonlocal ewm_mean, rolling_mean_short, rolling_mean_long, current_frame_number, lifetime_frame_number

        if not hass.data[DOMAIN]["active"]:
            return

        # Get the image from the camera entity
        camera = hass.data[DOMAIN]["camera"]
        try:
            image = await camera.async_camera_image()
        except Exception as e:
            LOGGER.error("Failed to get image from camera: %s", e)
            return

        # Encode the image to base64
        try:
            image_base64 = base64.b64encode(image).decode('utf-8')
        except Exception as e:
            LOGGER.error("Failed to encode image to base64: %s", e)
            return

        # Prepare the payload for the POST request
        payload = {
            "img": image_base64,
            "threshold": 0.08,
            "rectangleColor": [0, 0, 255],
            "rectangleThickness": 5,
            "fontFace": "FONT_HERSHEY_SIMPLEX",
            "fontScale": 1.5,
            "textColor": [0, 0, 255],
            "textThickness": 2
        }

        # Make the API call to the Obico ML server
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{obico_ml_api_host}/detect/", json=payload, headers={"Authorization": f"Bearer {obico_ml_api_token}"}) as response:
                    result = await response.json()
            except aiohttp.ClientError as e:
                LOGGER.error("Failed to make API call to Obico ML server: %s", e)
                return
            except Exception as e:
                LOGGER.error("Unexpected error: %s", e)
                return

            # Process the result
            p_sum = sum(detection[1] for detection in result["detections"])
            current_frame_number += 1
            lifetime_frame_number += 1

            ewm_mean = (p_sum * 2 / (12 + 1)) + (ewm_mean * (1 - 2 / (12 + 1)))
            rolling_mean_short = rolling_mean_short + ((p_sum - rolling_mean_short) / (310 if 310 <= current_frame_number else current_frame_number + 1))
            rolling_mean_long = rolling_mean_long + ((p_sum - rolling_mean_long) / (7200 if 7200 <= lifetime_frame_number else lifetime_frame_number + 1))

            adjusted_ewm_mean = 0  # Initialize adjusted_ewm_mean
            thresh_warning = 0  # Initialize thresh_warning
            thresh_failure = 0  # Initialize thresh_failure
            normalized_p = 0  # Initialize normalized_p

            if current_frame_number >= 3:
                # Calculate adjusted_ewm_mean, thresh_warning, thresh_failure, normalized_p
                pass

            # Update the entities based on the result
            hass.states.async_set(f"sensor.{device_name}_spaghetti_detection_failure_detection_result", result["detections"])
            hass.states.async_set(f"switch.{device_name}_spaghetti_detection_active", hass.data[DOMAIN]["active"])
            hass.states.async_set(f"camera.{device_name}_spaghetti_detection_camera", image_with_detections)

    # Track the time interval for the spaghetti detection handler
    hass.data[DOMAIN]["active"] = False
    hass.data[DOMAIN]["camera_entity_id"] = camera_entity_id
    hass.data[DOMAIN]["camera"] = None
    hass.data[DOMAIN]["update_interval"] = async_track_time_interval(hass, spaghetti_detection_handler, timedelta(seconds=update_interval))

    # Load platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    hass.data[DOMAIN]["device_name"] = device_name

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload the Spaghetti Detection integration."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN]["update_interval"]()
        hass.data[DOMAIN].pop(entry.entry_id)

    return unload_ok