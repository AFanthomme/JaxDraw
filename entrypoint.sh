#!/bin/bash
# Exit immediately if any command fails
set -e

# Run the device check
python tests/verify_devices.py

# Execute the main container command (this passes through CMD from Dockerfile or docker run)
exec "$@"