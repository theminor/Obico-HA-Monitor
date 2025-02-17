from homeassistant.components.camera import Camera, async_get_image
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from io import BytesIO
from . import DOMAIN

async def async_setup_entry(hass, entry, async_add_entities):
    device_name = hass.data[DOMAIN]["device_name"]
    camera_entity_id = hass.data[DOMAIN]["camera_entity_id"]
    camera = SpaghettiDetectionCamera(hass, f"{device_name} Spaghetti Detection Camera", entry.entry_id, camera_entity_id, device_name)
    hass.data[DOMAIN]["camera"] = camera
    async_add_entities([camera])

def draw_bounding_boxes(image, detections, rectangleColor=(255, 0, 0), rectangleThickness=5, fontSize=20, textColor=(255, 0, 0)):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", fontSize)
    for detection in detections:
        label, confidence, bbox = detection
        x, y, w, h = [int(v) for v in bbox]
        draw.rectangle([x, y, x + w, y + h], outline=rectangleColor, width=rectangleThickness)
        text = f"{label}: {confidence:.2f}"
        draw.text((x, y - fontSize), text, fill=textColor, font=font)
    return image

class SpaghettiDetectionCamera(Camera):
    def __init__(self, hass, name, entry_id, camera_entity_id, device_name):
        super().__init__()
        self.hass = hass
        self._name = name
        self._entry_id = entry_id
        self._camera_entity_id = camera_entity_id
        self._device_name = device_name
        self._image_data = None
        self._detections = []
        self._detection_result = {}

    @property
    def name(self):
        return self._name

    @property
    def state(self):
        return "idle"

    @property
    def unique_id(self):
        return f"{self._device_name}_spaghetti_detection_camera"

    async def async_camera_image(self):
        if self._image_data:
            image = Image.open(BytesIO(self._image_data))
            if self._detections:
                image = draw_bounding_boxes(image, self._detections)
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            return buffer.getvalue()

        # Fetch the image from the selected camera entity
        image = await async_get_image(self.hass, self._camera_entity_id)
        return image.content

    def update_image(self, image_data):
        self._image_data = image_data

    def update_detections(self, detections):
        self._detections = detections

    def update_detection_result(self, result):
        self._detection_result = result
        self._detections = result.get("result", {}).get("detections", [])

    async def async_update(self):
        # No need to update anything here
        pass