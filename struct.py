#!/usr/bin/env python3

"""
Demonstration script: 
- Simulates an EPICS-like data structure with 'display', 'description', etc.
- Uses a recursive function to convert nested dicts into frozensets,
  which are hashable and can serve as keys in a Python dictionary.
"""

def make_hashable(obj):
    """
    Recursively convert a nested data structure of dicts/lists/tuples/sets
    into an immutable, hashable form (using frozenset/tuple).
    This allows us to store it as a key in a dictionary.
    """
    if isinstance(obj, dict):
        # Convert each (key, value) to (key, make_hashable(value)),
        # then turn that set of pairs into a frozenset.
        return frozenset((k, make_hashable(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        # Convert each element recursively, return as a tuple (which is hashable).
        return tuple(make_hashable(x) for x in obj)
    else:
        # For strings, ints, floats, etc., just return as is.
        return obj

def ble_quadrupoles():
    """
    Simulates returning:
      - quad_dict: {quad_name -> channel_data (nested dict)}
      - quad_ptr:  {hashable(nested dict) -> quad_name}
    The nested dict includes a 'display' sub-dict with 'description'.
    """
    quadrupole_dict = {}
    quadrupole_ptr = {}

    # Some example quadrupoles
    quadrupole_list = [
        "QD01D", "QD01F", "QD02D", "QD02F",
        "QT03D1","QT03F2","QT03D3",
    ]

    for i, quad in enumerate(quadrupole_list):
        # Simulate a nested data structure
        fake_channel_data = {
            "value": 12.0 + i,
            "alarm": {"status": 0, "severity": 0},
            "timeStamp": 1678901234 + i,
            "display": {
                "limitLow": 0,
                "limitHigh": 10,
                "units": "T/m",
                "description": f"{quad}: field gradient"
            }
        }

        # Store the raw dict in quad_dict for forward lookup
        quadrupole_dict[quad] = fake_channel_data

        # Convert the nested dict to a hashable structure (frozensets)
        channel_data_key = make_hashable(fake_channel_data)
        # Reverse mapping: channel data (as a frozenset) -> quad name
        quadrupole_ptr[channel_data_key] = quad

    return quadrupole_dict, quadrupole_ptr

def copyPV(pv):
    """
    Mimics the callback that prints the nested data structure,
    specifically demonstrating how to look inside 'display' -> 'description'.
    """
    print("=== copyPV callback triggered ===")
    print("Full data structure:", pv)

    # Check for a top-level 'display' key
    if "display" in pv:
        display_section = pv["display"]
        # Then check for 'description' under 'display'
        if "description" in display_section:
            desc = display_section["description"]
            print(f"Description field: {desc}")

            # If you only want the part before the colon, e.g. "QD01D"
            quad_name = desc.split(":")[0].strip()
            print(f"Extracted quad name: {quad_name}")
        else:
            print("No 'description' key in the 'display' sub-dict.")
    else:
        print("No 'display' key at the top level of pv.")

    print()

def main():
    # 1) Get the nested dictionary structures
    quad_dict, quad_ptr = ble_quadrupoles()

    print("=== Forward mapping (quad_dict) ===")
    for quad_name, channel_data in quad_dict.items():
        print(f"{quad_name} -> {channel_data}")
    print()

    print("=== Reverse mapping (quad_ptr) ===")
    for channel_data_key, quad_name in quad_ptr.items():
        print(f"{channel_data_key} -> {quad_name}")
    print()

    # 2) Simulate 'subscribing' to each channel by calling copyPV ourselves
    for quad_name, channel_data in quad_dict.items():
        copyPV(channel_data)

if __name__ == "__main__":
    main()
