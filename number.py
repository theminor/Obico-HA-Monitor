from homeassistant.components.number import NumberEntity, NumberEntityDescription
from .const import DOMAIN

NUMBER_TYPES: tuple[NumberEntityDescription, ...] = (
    NumberEntityDescription(
        key="adjusted_ewm_mean",
        name="Adjusted EWM Mean",
    ),
    NumberEntityDescription(
        key="thresh_warning",
        name="Warning Threshold",
    ),
    NumberEntityDescription(
        key="thresh_failure",
        name="Failure Threshold",
    ),
    NumberEntityDescription(
        key="normalized_p",
        name="Confidence (Normalized P Value)",
    ),
)

async def async_setup_entry(hass, entry, async_add_entities):
    device_name = hass.data[DOMAIN]["device_name"]
    entities = [
        SpaghettiDetectionNumberEntity(
            NumberEntityDescription(
                key=entity_description.key,
                name=f"{device_name} {entity_description.name}",
                native_min_value=-1000000000000000,
                native_max_value=10000000000000000,
                native_step=0.000000001
            ),
            device_name
        ) for entity_description in NUMBER_TYPES
    ]

    async_add_entities(entities)

class SpaghettiDetectionNumberEntity(NumberEntity):
    def __init__(self, entity_description, device_name):
        self.entity_description = entity_description
        self.entity_id = f"number.{device_name}_spaghetti_detection_{entity_description.key}"
        self._attr_unique_id = f"number.{device_name}_spaghetti_detection_{entity_description.key}"
        if self._attr_native_value is None:
            self._attr_native_value = 0

    async def async_set_native_value(self, value: float) -> None:
        """Set the value of the number entity."""
        self._attr_native_value = value
        self.async_write_ha_state()