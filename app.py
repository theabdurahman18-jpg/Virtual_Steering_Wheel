"""
Easy launcher — double-click or run:  python app.py
"""

from main import run

if __name__ == "__main__":
    run()
import pyautogui
if direction == "LEFT":
    pyautogui.keyDown("left")
    pyautogui.keyUp("left")

elif direction == "RIGHT":
    pyautogui.keyDown("right")
    pyautogui.keyUp("right")