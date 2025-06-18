#!/bin/bash

# KCVM USB Gadget Setup Script
# Configures Raspberry Pi 5 as a USB HID device (keyboard + mouse)

set -e

GADGET_NAME="kcvm_hid"
VENDOR_ID="0x1d6b"  # Linux Foundation
PRODUCT_ID="0x0104" # Multifunction Composite Gadget
SERIAL="fedcba9876543210"
MANUFACTURER="KCVM Project"
PRODUCT="KCVM HID Device"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "This script must be run as root (use sudo)"
    exit 1
fi

# Load necessary modules
echo "Loading USB gadget modules..."
modprobe libcomposite

# Remove gadget if it exists
if [ -d "/sys/kernel/config/usb_gadget/$GADGET_NAME" ]; then
    echo "Removing existing gadget..."
    echo "" > /sys/kernel/config/usb_gadget/$GADGET_NAME/UDC
    rm -rf /sys/kernel/config/usb_gadget/$GADGET_NAME
fi

# Create gadget
echo "Creating USB gadget..."
mkdir -p /sys/kernel/config/usb_gadget/$GADGET_NAME
cd /sys/kernel/config/usb_gadget/$GADGET_NAME

# Set vendor and product IDs
echo $VENDOR_ID > idVendor
echo $PRODUCT_ID > idProduct

# Set device version and USB version
echo 0x0100 > bcdDevice  # v1.0.0
echo 0x0200 > bcdUSB     # USB 2.0

# Set device class (0x00 for composite device)
echo 0x00 > bDeviceClass
echo 0x00 > bDeviceSubClass
echo 0x00 > bDeviceProtocol

# Set max packet size
echo 0x40 > bMaxPacketSize0

# Create English strings
mkdir -p strings/0x409
echo $SERIAL > strings/0x409/serialnumber
echo $MANUFACTURER > strings/0x409/manufacturer
echo $PRODUCT > strings/0x409/product

# Create configuration
mkdir -p configs/c.1
echo 250 > configs/c.1/MaxPower  # 250mA
echo 0x80 > configs/c.1/bmAttributes  # Bus powered

# Create configuration strings
mkdir -p configs/c.1/strings/0x409
echo "HID Configuration" > configs/c.1/strings/0x409/configuration

# Create HID functions
echo "Creating HID functions..."

# Mouse HID function
mkdir -p functions/hid.0
echo 0 > functions/hid.0/protocol      # Boot protocol
echo 0 > functions/hid.0/subclass      # No subclass
echo 5 > functions/hid.0/report_length  # 5-byte reports

# Mouse HID report descriptor
# Standard mouse with 3 buttons, X/Y movement, and wheel
echo -ne '\x05\x01\x09\x02\xa1\x01\x09\x01\xa1\x00\x05\x09\x19\x01\x29\x03\x15\x00\x25\x01\x95\x03\x75\x01\x81\x02\x95\x01\x75\x05\x81\x03\x05\x01\x09\x30\x09\x31\x09\x38\x15\x81\x25\x7f\x75\x08\x95\x03\x81\x06\xc0\xc0' > functions/hid.0/report_desc

# Keyboard HID function  
mkdir -p functions/hid.1
echo 1 > functions/hid.1/protocol      # Keyboard protocol
echo 1 > functions/hid.1/subclass      # Boot interface subclass
echo 8 > functions/hid.1/report_length  # 8-byte reports

# Keyboard HID report descriptor
# Standard keyboard with modifier keys and 6-key rollover
echo -ne '\x05\x01\x09\x06\xa1\x01\x05\x07\x19\xe0\x29\xe7\x15\x00\x25\x01\x75\x01\x95\x08\x81\x02\x95\x01\x75\x08\x81\x03\x95\x05\x75\x01\x05\x08\x19\x01\x29\x05\x91\x02\x95\x01\x75\x03\x91\x03\x95\x06\x75\x08\x15\x00\x25\x65\x05\x07\x19\x00\x29\x65\x81\x00\xc0' > functions/hid.1/report_desc

# Link functions to configuration
echo "Linking HID functions to configuration..."
ln -s functions/hid.0 configs/c.1/
ln -s functions/hid.1 configs/c.1/

# Enable gadget
echo "Enabling USB gadget..."
UDC_DEVICE=$(ls /sys/class/udc | head -n 1)

if [ -z "$UDC_DEVICE" ]; then
    echo "Error: No UDC device found!"
    echo "Make sure dwc2 module is loaded"
    exit 1
fi

echo $UDC_DEVICE > UDC

# Create device nodes if they don't exist
echo "Creating device nodes..."
if [ ! -e /dev/hidg0 ]; then
    major=$(grep hid /proc/devices | cut -d' ' -f1)
    mknod /dev/hidg0 c $major 0
    chmod 666 /dev/hidg0
fi

if [ ! -e /dev/hidg1 ]; then
    major=$(grep hid /proc/devices | cut -d' ' -f1)  
    mknod /dev/hidg1 c $major 1
    chmod 666 /dev/hidg1
fi

echo "USB gadget setup complete!"
echo "Mouse device: /dev/hidg0"
echo "Keyboard device: /dev/hidg1"
echo ""
echo "Test with:"
echo "  Mouse:    echo -ne '\\x00\\x10\\x10\\x00\\x00' > /dev/hidg0"
echo "  Keyboard: echo -ne '\\x00\\x00\\x04\\x00\\x00\\x00\\x00\\x00' > /dev/hidg1"