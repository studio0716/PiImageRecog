# KCVM - Keyboard, Camera, Video Mouse

A Raspberry Pi 5 application that uses AI-powered computer vision to control a computer's mouse and keyboard by visually monitoring the screen.

## Overview

KCVM turns your Raspberry Pi 5 with AI HAT and AI Camera into a smart USB input device. It watches your computer screen through the camera and can control the mouse cursor and keyboard inputs, making it useful for:

- Remote computer control without software installation on the target
- Accessibility applications
- Automated testing of visual interfaces
- Computer control in secure environments

## Hardware Requirements

- Raspberry Pi 5 (4GB or 8GB recommended)
- Raspberry Pi AI HAT
- Raspberry Pi AI Camera Module
- USB-C cable (for data connection to target computer)
- Power supply for Raspberry Pi
- Camera mount or tripod

## Features

- Real-time screen detection and perspective correction
- Mouse cursor tracking and movement control
- Keyboard input emulation
- Automatic screen calibration
- Low-latency video processing using AI HAT acceleration
- USB HID device emulation (no drivers needed on target computer)

## Quick Start

1. **Hardware Setup**: Follow the [hardware setup guide](docs/hardware_setup.md)
2. **Software Installation**: See [software setup guide](docs/software_setup.md)
3. **First Run**: Check the [usage guide](docs/usage.md)

## Project Status

This project is currently in active development. Core features being implemented:

- [x] Project architecture design
- [ ] USB HID device configuration
- [ ] Computer vision pipeline
- [ ] Screen detection and calibration
- [ ] Mouse cursor tracking
- [ ] Keyboard input handling
- [ ] Real-time performance optimization

## License

MIT License - See LICENSE file for details