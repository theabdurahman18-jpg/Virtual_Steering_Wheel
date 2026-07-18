# Virtual Steering Wheel

Control keyboard steering in games and apps using your webcam and both hands. This project detects hand landmarks with **MediaPipe**, computes the angle between your hands to simulate a steering wheel, and sends **Left/Right Arrow** key input via **PyAutoGUI**.

## Features

- Real-time dual-hand detection (index finger, thumb, wrist)
- Steering angle from the line between both hands
- Moving average smoothing for stable control
- On-screen virtual steering wheel and HUD (FPS, hands, angle, direction)
- Keyboard mapping:
  - **Angle < -15°** → hold **Left Arrow**
  - **Angle > 15°** → hold **Right Arrow**
  - **-15° to 15°** → release keys (straight)
- Graceful handling when one or both hands are not visible

## Project Structure

```
Virtual_Steering_Wheel/
├── main.py                 # Application entry point and main loop
├── hand_tracker.py         # MediaPipe hand detection and landmarks
├── steering.py             # Angle math, smoothing, wheel overlay
├── keyboard_controller.py  # PyAutoGUI arrow key press/release
├── utils.py                # FPS counter, helpers, HUD text
├── requirements.txt        # Pinned dependency versions
└── README.md
```

## Requirements

- Python 3.9–3.11 (recommended)
- Webcam
- Windows, macOS, or Linux

## Installation

1. **Clone or download this project**

   ```bash
   git clone https://github.com/jayesh-cmd/virtual-steering-wheel.git
   cd virtual-steering-wheel
   ```

   Or use this folder directly if you already have the source.

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Dependency Versions

| Package        | Version   |
|----------------|-----------|
| opencv-python  | 4.10.0.84 |
| mediapipe      | 0.10.14   |
| pyautogui      | 0.9.54    |
| numpy          | 1.26.4    |

## Usage

1. Run the application:

   ```bash
   python main.py
   ```

2. Optional arguments:

   ```bash
   python main.py --camera 0 --width 1280 --height 720
   ```

3. Hold both hands in front of the camera as if gripping a steering wheel (shoulder-width apart).

4. Rotate your hands:
   - Tilt so the **left hand is higher** → turn **left**
   - Tilt so the **right hand is higher** → turn **right**
   - Keep hands level → **straight** (keys released)

5. Press **Q** or **ESC** to quit.

## How It Works

1. **Hand tracking** — MediaPipe detects up to two hands and reads wrist, thumb tip, and index tip landmarks.
2. **Angle calculation** — The angle of the line between left and right hand centers (relative to horizontal) represents wheel rotation.
3. **Smoothing** — A moving average over recent frames reduces jitter.
4. **Direction mapping** — Angles outside ±15° trigger left/right arrow keys; inside the dead zone keys are released.
5. **Visualization** — A circular overlay rotates with the smoothed angle; HUD shows FPS and status.

## Performance Tips (30+ FPS)

- Use default resolution (`1280x720`) or lower (`--width 960 --height 540`) if needed
- Ensure good lighting and a clear background
- `model_complexity=0` is used by default for faster inference
- Close other apps using the webcam

## Troubleshooting

| Issue | Suggestion |
|-------|------------|
| Camera not opening | Try `--camera 1` or check webcam permissions |
| Low FPS | Lower resolution; improve lighting |
| Keys not affecting game | Run game and app with same focus rules; some games block synthetic input |
| Wrong left/right | Mirror mode is enabled; swap hand positions or adjust your posture |

## Safety

PyAutoGUI **FAILSAFE** is enabled: move the mouse to the top-left corner of the screen to abort PyAutoGUI actions if needed.

## License

MIT — use and modify freely.
