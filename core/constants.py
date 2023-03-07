from enum import Enum


ACTIVE_STATE_VISITATION_KEY = "active_state_visitation"
UNCERTAINTY_LOSS_KEY = "uncertainty_loss"
CL_ENV_KEYS = ["cold_Texas", "dry_Cali", "hot_new_york", "snowy_Cali_winter"]
DEFAULT_REW_MAP = {
    b'E': -0.2,
    b'S': -0.2,
    b'W': -5.0,
    b'G': 10.0,
    b'U': -0.2,
    b'R': -0.2,
    b"L": -0.2,
    b"D": -0.2,
    b"B": -10.0
}

RESPAWNABLE_TOKENS = [".", "P"]
class Environments(Enum):
    GRIDWORLD = "gw"
    CITYLEARN = "cl"
    DM_MAZE = "dm"
    SINERGYM = "sg"

class SG_WEATHER_TYPES(Enum):
    HOT = "hot"
    COOL = "cool"
    MIXED = "mixed"