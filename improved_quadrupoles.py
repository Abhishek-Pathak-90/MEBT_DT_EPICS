#!/usr/bin/python3

try:
    import pvaccess
except ImportError:
    raise ImportError("pvaccess module not found. Please ensure it is installed.")

def ble_quadrupoles():
    """
    Creates PV access channels for MEBT BLE quadrupoles.
    
    Returns:
        tuple: A pair of dictionaries:
            - quadrupole_channels: Maps quadrupole names to their PV channel objects
            - channel_to_quad_map: Maps PV channel objects back to quadrupole names
    """
    quadrupole_channels = {}
    channel_to_quad_map = {}
    quadrupole_list = [
        "QD01D", "QD01F", "QD02D", "QD02F",
        "QT03D1", "QT03F2", "QT03D3",
        "QT04D1", "QT04F2", "QT04D3",
        "QT05D1", "QT05F2", "QT05D3",
        "QT06D1", "QT06F2", "QT06D3",
        "QT07D1", "QT07F2", "QT07D3",
        "QT08D1", "QT08F2", "QT08D3",
        "QT09D1", "QT09F2", "QT09D3",
        "QT10D1", "QT10F2", "QT10D3",
        "QT11D1", "QT11F2", "QT11D3"
    ]

    for quad in quadrupole_list:
        try:
            channel = pvaccess.Channel(f"WFE:MEBT_BLE_{quad}:Gradient")
            quadrupole_channels[quad] = channel
            channel_to_quad_map[channel] = quad
        except Exception as e:
            print(f"Failed to create channel for quadrupole {quad}: {str(e)}")
            continue

    return quadrupole_channels, channel_to_quad_map
