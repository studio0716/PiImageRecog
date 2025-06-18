# Hardware Setup Guide

This guide will walk you through assembling and connecting the hardware components for the KCVM system.

## Required Components

### Essential Hardware
1. **Raspberry Pi 5** (4GB or 8GB model)
2. **Raspberry Pi AI HAT** (M.2 HAT+ with Hailo-8L AI accelerator)
3. **Raspberry Pi AI Camera** (IMX708-based 12MP camera module)
4. **USB-C to USB-A Cable** (for connecting to target computer)
5. **Power Supply** (27W USB-C PD power adapter recommended)
6. **MicroSD Card** (32GB or larger, Class 10/A1)

### Optional but Recommended
- Camera mount or flexible gooseneck holder
- Case for Raspberry Pi 5 (compatible with AI HAT)
- Heat sinks or active cooling

## Assembly Instructions

### Step 1: Prepare the Raspberry Pi 5

1. If using heat sinks, attach them to the CPU and other chips now
2. Insert the microSD card (we'll flash the OS later)
3. Do not connect power yet

### Step 2: Install the AI HAT

1. Ensure the Raspberry Pi is powered off
2. Locate the M.2 connector on the Pi 5 board
3. Carefully align the AI HAT with the M.2 connector
4. Insert at a 30-degree angle, then press down
5. Secure with the provided mounting screw

### Step 3: Connect the AI Camera

1. Locate the CAM/DISP 0 port on the Raspberry Pi 5
2. Gently lift the connector's locking tab
3. Insert the camera ribbon cable with the contacts facing the HDMI ports
4. Press down the locking tab to secure the cable
5. Mount the camera module in your chosen holder

### Step 4: Connection Diagram

```
[AI Camera] ----ribbon----> [CAM/DISP 0 Port]
                                    |
                            [Raspberry Pi 5]
                                    |
                              [AI HAT M.2]
                                    |
                            [USB-C Data Port] ----USB cable----> [Target Computer]
                                    |
                            [USB-C Power Port] <---- [27W Power Supply]
```

### Step 5: Initial Power Test

1. Connect the power supply to the POWER USB-C port
2. The Pi should boot (green LED activity)
3. If no activity, check all connections

## Camera Positioning

For optimal performance:

1. **Distance**: Position camera 30-50cm from the monitor
2. **Angle**: Camera should face the screen directly (minimize angle)
3. **Stability**: Use a sturdy mount to prevent vibration
4. **Lighting**: Avoid glare on the screen; indirect lighting works best

## USB Connection Notes

- The Raspberry Pi 5 has two USB-C ports:
  - **Left port**: Power only (use for power supply)
  - **Right port**: Data + Power (use for computer connection)
- When connected to a computer, the Pi can draw limited power
- Always use the dedicated power supply for reliable operation

## Troubleshooting

### Pi Won't Boot
- Check power supply (needs 5V/5A capability)
- Verify microSD card is properly inserted
- Try without AI HAT first to isolate issues

### Camera Not Detected
- Reseat the ribbon cable
- Check cable orientation (contacts toward HDMI)
- Ensure locking tab is fully closed

### AI HAT Not Recognized
- Verify M.2 connection is secure
- Check for bent pins
- Update Pi firmware (we'll do this in software setup)

## Next Steps

Once hardware is assembled:
1. Proceed to [Software Setup](software_setup.md)
2. The software setup will configure the Pi as a USB gadget device
3. We'll install all necessary drivers and libraries