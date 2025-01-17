# MEBT Digital Twin EPICS Implementation

This repository contains the implementation of a Digital Twin for the Medium Energy Beam Transport (MEBT) system using EPICS (Experimental Physics and Industrial Control System).

## Features

- Real-time beam size monitoring and visualization
- Integration with EPICS control system
- BPM (Beam Position Monitor) data processing
- Dynamic quadrupole gradient updates
- Interactive plotting with matplotlib

## Requirements

- Python 3.x
- EPICS
- numpy
- matplotlib
- watchdog
- pvaccess

## Setup and Installation

1. Clone the repository
2. Install the required dependencies
3. Configure EPICS environment variables
4. Run the main script

## Usage

```bash
python MEBT_BPM_RMS.py
```

## File Structure

- `MEBT_BPM_RMS.py`: Main implementation file
- `mebt_final.dat`: Lattice configuration
- `Individual_matrix.dat`: Transfer matrices
- `quad_gradients.txt`: Quadrupole gradient settings

## Contributing

Feel free to open issues or submit pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
