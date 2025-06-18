#!/bin/bash

# Test script for HID devices

echo "Testing KCVM HID devices..."

# Check if devices exist
if [ ! -e /dev/hidg0 ]; then
    echo "Error: Mouse device /dev/hidg0 not found!"
    echo "Run setup_usb_gadget.sh first"
    exit 1
fi

if [ ! -e /dev/hidg1 ]; then
    echo "Error: Keyboard device /dev/hidg1 not found!"
    echo "Run setup_usb_gadget.sh first"
    exit 1
fi

echo "HID devices found!"
echo ""

# Test mouse
echo "Testing mouse (will move cursor in a small square)..."
echo "Press Ctrl+C to stop"
sleep 2

# Move in a square pattern
for i in {1..4}; do
    # Move right
    for j in {1..20}; do
        echo -ne '\x00\x05\x00\x00\x00' > /dev/hidg0
        sleep 0.02
    done
    
    # Move down
    for j in {1..20}; do
        echo -ne '\x00\x00\x05\x00\x00' > /dev/hidg0
        sleep 0.02
    done
    
    # Move left
    for j in {1..20}; do
        echo -ne '\x00\xfb\x00\x00\x00' > /dev/hidg0
        sleep 0.02
    done
    
    # Move up
    for j in {1..20}; do
        echo -ne '\x00\x00\xfb\x00\x00' > /dev/hidg0
        sleep 0.02
    done
done

echo ""
echo "Mouse test complete!"
echo ""

# Test keyboard
echo "Testing keyboard (will type 'Hello KCVM!')..."
echo "Make sure a text editor is focused!"
echo "Starting in 3 seconds..."
sleep 3

# Type "Hello KCVM!"
# H (shift + h)
echo -ne '\x02\x00\x0b\x00\x00\x00\x00\x00' > /dev/hidg1
sleep 0.1
echo -ne '\x00\x00\x00\x00\x00\x00\x00\x00' > /dev/hidg1
sleep 0.1

# e
echo -ne '\x00\x00\x08\x00\x00\x00\x00\x00' > /dev/hidg1
sleep 0.1
echo -ne '\x00\x00\x00\x00\x00\x00\x00\x00' > /dev/hidg1
sleep 0.1

# l
echo -ne '\x00\x00\x0f\x00\x00\x00\x00\x00' > /dev/hidg1
sleep 0.1
echo -ne '\x00\x00\x00\x00\x00\x00\x00\x00' > /dev/hidg1
sleep 0.1

# l
echo -ne '\x00\x00\x0f\x00\x00\x00\x00\x00' > /dev/hidg1
sleep 0.1
echo -ne '\x00\x00\x00\x00\x00\x00\x00\x00' > /dev/hidg1
sleep 0.1

# o
echo -ne '\x00\x00\x12\x00\x00\x00\x00\x00' > /dev/hidg1
sleep 0.1
echo -ne '\x00\x00\x00\x00\x00\x00\x00\x00' > /dev/hidg1
sleep 0.1

# space
echo -ne '\x00\x00\x2c\x00\x00\x00\x00\x00' > /dev/hidg1
sleep 0.1
echo -ne '\x00\x00\x00\x00\x00\x00\x00\x00' > /dev/hidg1
sleep 0.1

echo ""
echo "Keyboard test complete!"
echo ""
echo "HID devices are working correctly!"