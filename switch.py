from homeassistant.components.switch import SwitchEntity
from . import DOMAIN

async def async_setup_entry(hass, entry, async_add_entities):
    device_name = hass.data[DOMAIN]["device_name"]
    switch = SpaghettiDetectionSwitch(hass, f"{device_name} Spaghetti Detection Active", entry.entry_id, device_name)
    async_add_entities([switch])

class SpaghettiDetectionSwitch(SwitchEntity):
    def __init__(self, hass, name, entry_id, device_name):
        self.hass = hass
        self._name = name
        self._entry_id = entry_id
        self._device_name = device_name
        self._is_on = False

    @property
    def name(self):
        return self._name

    @property
    def is_on(self):
        return self._is_on

    async def async_turn_on(self, **kwargs):
        self._is_on = True
        self.hass.data[DOMAIN]["active"] = True
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs):
        self._is_on = False
        self.hass.data[DOMAIN]["active"] = False
        self.async_write_ha_state()

    @property
    def unique_id(self):
        return f"{self._device_name}_spaghetti_detection_switch"