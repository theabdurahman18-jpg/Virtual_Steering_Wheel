"""
Virtual Steering Wheel - main application entry point.

Uses webcam hand tracking to simulate steering wheel input via keyboard arrows.
Includes a built-in test window and live demo bar — no Notepad required.
"""

import argparse
import time

import cv2

from demo_panel import SteeringDemoPanel
from hand_tracker import HandTracker
from keyboard_controller import KeyboardController
from steering import SteeringController, SteeringDirection, SteeringWheelRenderer
from test_window import SteeringTestWindow
from utils import FpsCounter, draw_text_block


def parse_args() -> argparse.Namespace:
    """Parse command-line options for camera index and display settings."""
    parser = argparse.ArgumentParser(description="Virtual Steering Wheel")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Capture frame width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Capture frame height (default: 720)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=6.0,
        help="Degrees of tilt needed to turn (default: 6)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print key press events in the terminal",
    )
    parser.add_argument(
        "--no-test-window",
        action="store_true",
        help="Disable the built-in steering test text window",
    )
    parser.add_argument(
        "--no-keyboard",
        action="store_true",
        help="Disable arrow-key output (demo mode only)",
    )
    return parser.parse_args()


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    """
    Open the default webcam with DirectShow on Windows for lower latency.

    Raises RuntimeError when the camera cannot be opened.
    """
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open camera index {camera_index}. "
            "Check that a webcam is connected and not in use."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, 30)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def build_hud_lines(
    fps: float,
    num_hands: int,
    angle: float,
    direction: SteeringDirection,
    hands_ready: bool,
    active_key: str,
    threshold: float,
) -> list[str]:
    """Compose HUD text lines shown on the webcam feed."""
    if num_hands == 0:
        hand_hint = "Raise BOTH hands in front of camera"
    elif num_hands == 1:
        hand_hint = "1 hand OK — tilt it up/down to steer"
    elif direction == SteeringDirection.STRAIGHT:
        hand_hint = f"Tilt one hand UP (need > {threshold:.0f} deg)"
    else:
        hand_hint = "Steering ACTIVE"

    return [
        f"FPS: {fps:.1f}",
        f"Hands detected: {num_hands}/2",
        f"Steering angle: {angle:+.1f} deg",
        f"Direction: {SteeringController.direction_label(direction)}",
        f"Keyboard: {active_key}",
        hand_hint,
        "Watch orange car at bottom + Steering Test Window",
        "Press Q or ESC to quit",
    ]


def run() -> None:
    """Main loop: capture frames, track hands, steer, and render overlays."""
    args = parse_args()
    capture = open_camera(args.camera, args.width, args.height)

    hand_tracker = HandTracker(model_complexity=0)
    steering = SteeringController(
        smoothing_window=6,
        straight_threshold_deg=args.threshold,
    )
    keyboard = None if args.no_keyboard else KeyboardController()
    test_window = None if args.no_test_window else SteeringTestWindow()
    demo_panel = SteeringDemoPanel()
    wheel_renderer = SteeringWheelRenderer(
        center=(args.width // 2, 120),
        straight_threshold_deg=args.threshold,
    )
    fps_counter = FpsCounter(window_size=30)

    window_name = "Virtual Steering Wheel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Virtual Steering Wheel started.")
    print("  - A 'Steering Test Window' opens — cursor moves when you steer.")
    print("  - Orange car at bottom of camera feed also moves.")
    print("  - Tilt one hand UP (two hands) or tilt single hand up/down.")
    print(f"  - Turn triggers past +/- {args.threshold:.0f} degrees.")
    if args.debug:
        print("  Debug mode: keyboard events print here.")

    try:
        while True:
            loop_start = time.perf_counter()
            success, frame = capture.read()
            if not success:
                print("Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)

            annotated, hands = hand_tracker.process_frame(frame)
            hand_map = hand_tracker.summarize_detection(hands)

            angle, direction, hands_ready = steering.update(
                hand_map["left"],
                hand_map["right"],
            )

            if keyboard is not None:
                keyboard.update(direction, debug=args.debug)

            if test_window is not None:
                test_window.update(direction)

            wheel_renderer.draw(annotated, angle, direction, hands_ready)
            demo_panel.draw(annotated, direction, hands_ready)

            fps = fps_counter.tick(loop_start)
            key_label = keyboard.active_key_label() if keyboard else "disabled"
            hud_lines = build_hud_lines(
                fps=fps,
                num_hands=len(hands),
                angle=angle,
                direction=direction,
                hands_ready=hands_ready,
                active_key=key_label,
                threshold=args.threshold,
            )
            draw_text_block(annotated, hud_lines, origin=(10, 30))

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        if keyboard is not None:
            keyboard.shutdown()
        if test_window is not None:
            test_window.close()
        hand_tracker.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
