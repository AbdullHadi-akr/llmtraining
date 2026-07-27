def get_profiles(operation: str) -> tuple[bool, bool]:
    """
    Returns:
        (
            has_fluid_temperature_profile,
            has_volume_flow_profile
        )
    """

    profiles = {
        "OP1":  (False, False),
        "OP2":  (False, False),
        "OP3":  (False, False),
        "OP4":  (False, False),
        "OP5":  (False, False),
        "OP6":  (False, False),
        "OP7":  (False, False),
        "OP8":  (True,  False),
        "OP9":  (True,  False),
        "OP10": (False, False),
        "OP11": (False, False),
        "OP12": (True,  False),
        "OP13": (True,  False),
        "OP14": (False, False),
        "OP15": (True,  True),
        "OP16": (False, False),
    }

    return profiles[operation]