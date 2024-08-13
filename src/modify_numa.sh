#!/bin/bash
for a in /sys/bus/pci/devices/*; do
  echo 0 > "$a/numa_node" 2>/dev/null || true
done
