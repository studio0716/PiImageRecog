# Software Setup Guide

This guide covers installing and configuring all software components for KCVM.

## Prerequisites

- Completed [hardware setup](hardware_setup.md)
- Another computer for initial Pi setup
- Internet connection

## Step 1: Install Raspberry Pi OS

### Download and Flash OS

1. Download **Raspberry Pi Imager** from https://www.raspberrypi.com/software/
2. Insert your microSD card into your computer
3. Open Raspberry Pi Imager and select:
   - **Device**: Raspberry Pi 5
   - **OS**: Raspberry Pi OS (64-bit) - Full version
   - **Storage**: Your microSD card

### Configure OS Settings

Before writing, click the gear icon for settings:
- **Hostname**: kcvm
- **Enable SSH**: Yes (password authentication)
- **Username**: pi (or your preference)
- **Password**: Set a secure password
- **WiFi**: Configure your network (if using WiFi)
- **Locale**: Set your timezone

Write the image and wait for completion.

## Step 2: First Boot and Basic Setup

1. Insert the microSD card into your Pi
2. Connect power and let it boot (2-3 minutes)
3. SSH into your Pi:
   ```bash
   ssh pi@kcvm.local
   # Or use IP address if mDNS doesn't work
   ```

4. Update the system:
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo rpi-update  # Updates firmware for AI HAT support
   sudo reboot
   ```

## Step 3: Enable USB Gadget Mode

### Configure Device Tree

1. Edit boot configuration:
   ```bash
   sudo nano /boot/firmware/config.txt
   ```

2. Add at the end:
   ```
   # Enable USB Gadget mode
   dtoverlay=dwc2,dr_mode=peripheral
   
   # Enable AI HAT
   dtparam=pciex1_gen=3
   ```

3. Edit cmdline:
   ```bash
   sudo nano /boot/firmware/cmdline.txt
   ```

4. Add after `rootwait` (keep everything on one line):
   ```
   modules-load=dwc2,g_hid
   ```

### Install USB Gadget Tools

```bash
sudo apt install -y python3-pip python3-venv git
sudo apt install -y libusb-1.0-0-dev libudev-dev
```

## Step 4: Install KCVM Software

### Clone and Setup Repository

```bash
cd ~
git clone https://github.com/yourusername/KCVM.git
cd KCVM

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
```

### Run Setup Script

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run the setup script
sudo ./scripts/setup_usb_gadget.sh

# Install Python dependencies
./scripts/install_deps.sh
```

## Step 5: Configure AI HAT and Camera

### Install Hailo Software

```bash
# Add Hailo repository
wget -qO- https://hailo.ai/files/install.sh | sudo bash

# Install runtime and Python bindings
sudo apt install -y hailo-all

# Verify installation
hailortcli fw-control identify
```

### Enable Camera

```bash
# Enable camera interface
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable

# Install camera libraries
sudo apt install -y python3-picamera2 python3-opencv
```

## Step 6: Test Basic Functionality

### Test USB HID

```bash
# Load the test HID module
sudo modprobe g_hid
sudo ./scripts/test_hid.sh

# Check if device appears
ls /dev/hidg*
```

### Test Camera

```bash
python3 -c "from picamera2 import Picamera2; cam = Picamera2(); cam.start(); print('Camera OK'); cam.stop()"
```

### Test AI HAT

```bash
hailortcli run model_zoo/yolov5m.hef --input-files /dev/video0
```

## Step 7: Configure Auto-start

### Create SystemD Service

```bash
sudo nano /etc/systemd/system/kcvm.service
```

Add:
```ini
[Unit]
Description=KCVM - Keyboard Camera Video Mouse
After=multi-user.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/KCVM
Environment="PATH=/home/pi/KCVM/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStartPre=/home/pi/KCVM/scripts/setup_usb_gadget.sh
ExecStart=/home/pi/KCVM/venv/bin/python /home/pi/KCVM/src/main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable kcvm.service
```

## Troubleshooting

### USB Device Not Recognized

1. Check kernel modules:
   ```bash
   lsmod | grep dwc2
   lsmod | grep g_hid
   ```

2. Check USB gadget configuration:
   ```bash
   ls /sys/kernel/config/usb_gadget/
   ```

### Camera Issues

1. Verify camera connection:
   ```bash
   vcgencmd get_camera
   libcamera-hello --list-cameras
   ```

### AI HAT Not Working

1. Check PCIe connection:
   ```bash
   lspci | grep Hailo
   ```

2. Check dmesg for errors:
   ```bash
   dmesg | grep -i hailo
   ```

## Next Steps

1. Proceed to [Usage Guide](usage.md) to start using KCVM
2. Run initial calibration
3. Test with your target computer