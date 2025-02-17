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
    camera_entity_id = entry.data["camera_entity"]
    update_interval = entry.data.get("update_interval", 60)
    obico_ml_api_host = entry.data.get("obico_ml_api_host", "http://127.0.0.1:3333")
    obico_ml_api_token = entry.data.get("obico_ml_api_token", "obico_api_secret")
    printer_device = entry.data["printer_device"]
    device_name = entry.data["device_name"]

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
        except Exception as e:
            LOGGER.error("Failed to get image from camera: %s", e)
            return

        predictions = []
        last_prediction = PrinterPrediction()  # TO DO - need to implement - see obico-server/backend/app/models/other_models.py
        ### *** STOPPED REVIEW HERE *** ###

        jpg_filenames = sorted(os.listdir(jpgs_dir))
        for jpg_path in jpg_filenames:
            jpg_abs_path = os.path.join(jpgs_dir, jpg_path)
            with open(jpg_abs_path, 'rb') as pic:
                pic_path = f'{_print.user.id}/{_print.id}/{jpg_path}'
                internal_url, _ = save_file_obj(f'uploaded/{pic_path}', pic, settings.PICS_CONTAINER, _print.user.syndicate.name, long_term_storage=False)
                req = requests.get(settings.ML_API_HOST + '/p/', params={'img': internal_url}, headers=ml_api_auth_headers(), verify=False)
                req.raise_for_status()
                detections = req.json()['detections']
                update_prediction_with_detections(last_prediction, detections, _print.printer)
                predictions.append(last_prediction)

                if is_failing(last_prediction, 1, escalating_factor=1):
                    _print.alerted_at = timezone.now()

                last_prediction = copy.deepcopy(last_prediction)
                detections_to_visualize = [d for d in detections if d[1] > VISUALIZATION_THRESH]
                overlay_detections(Image.open(jpg_abs_path), detections_to_visualize).save(os.path.join(tagged_jpgs_dir, jpg_path), "JPEG")

        predictions_json = serializers.serialize("json", predictions)
        _, json_url = save_file_obj(f'private/{_print.id}_p.json', io.BytesIO(str.encode(predictions_json)), settings.TIMELAPSE_CONTAINER, _print.user.syndicate.name)

        mp4_filename = f'{_print.id}_tagged.mp4'
        output_mp4 = os.path.join(tmp_dir, mp4_filename)
        subprocess.run(
            f'ffmpeg -y -r 30 -pattern_type glob -i {tagged_jpgs_dir}/*.jpg -c:v libx264 -pix_fmt yuv420p -vf pad=ceil(iw/2)*2:ceil(ih/2)*2 {output_mp4}'.split(), check=True)
        with open(output_mp4, 'rb') as mp4_file:
            _, mp4_file_url = save_file_obj(f'private/{mp4_filename}', mp4_file, settings.TIMELAPSE_CONTAINER, _print.user.syndicate.name)

        with open(os.path.join(jpgs_dir, jpg_filenames[-1]), 'rb') as poster_file:
            _, poster_file_url = save_file_obj(f'private/{_print.id}_poster.jpg', poster_file, settings.TIMELAPSE_CONTAINER, _print.user.syndicate.name)

        _print.tagged_video_url = mp4_file_url
        _print.prediction_json_url = json_url
        _print.poster_url = poster_file_url
        _print.save(keep_deleted=True)

        shutil.rmtree(tmp_dir, ignore_errors=True)
        send_timelapse_detection_done_email(_print)
        delete_dir(f'uploaded/{_print.user.id}/{_print.id}/', settings.PICS_CONTAINER, long_term_storage=False)
    ################














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