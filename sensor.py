from homeassistant.components.sensor import SensorEntity
from . import DOMAIN

async def async_setup_entry(hass, entry, async_add_entities):
    device_name = hass.data[DOMAIN]["device_name"]
    sensor = FailureDetectionSensor(hass, f"{device_name} Spaghetti Detection State", entry.entry_id, device_name)
    async_add_entities([sensor])

class FailureDetectionSensor(SensorEntity):
    def __init__(self, hass, name, entry_id, device_name):
        self.hass = hass
        self._name = name
        self._entry_id = entry_id
        self._device_name = device_name
        self._state = "Unknown"

    @property
    def name(self):
        return self._name

    @property
    def state(self):
        return self._state

    @property
    def unique_id(self):
        return f"{self._device_name}_spaghetti_detection_state"

    def update_state(self, new_state):
        self._state = new_state
        self.async_write_ha_state()