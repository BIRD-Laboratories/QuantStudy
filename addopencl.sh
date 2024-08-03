#!/bin/bash

# Update and install necessary packages
echo "Updating package list and installing necessary packages..."
apt-get update
apt-get install -y cmake build-essential ocl-icd-opencl-dev clinfo beignet-opencl-icd

# Verify OpenCL installation
echo "Verifying OpenCL installation..."
clinfo

echo "OpenCL support for CPU has been added to your Google Colab environment."