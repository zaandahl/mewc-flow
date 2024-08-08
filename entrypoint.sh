#!/bin/bash

# Modify NUMA node settings
for a in /sys/bus/pci/devices/*; do
  echo 0 > "$a/numa_node" 2>/dev/null || true
done

# Run the main container process
exec "$@"
