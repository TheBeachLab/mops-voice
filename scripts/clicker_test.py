"""Tiny diagnostic: print every key press/release until Ctrl-C.

Used to map an unknown HID device (e.g. a BT presentation remote) to
the keys it emits, so we can wire those keys into mops-voice's
push-to-talk listener.

Run from the repo root:
    uv run python scripts/clicker_test.py
"""

from pynput import keyboard


def on_press(key):
    print(f"press    {key!r}")


def on_release(key):
    print(f"release  {key!r}")
    if key == keyboard.Key.esc:
        return False  # stop listener on Esc


with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    print("Click each clicker button. Press Esc on the keyboard to exit.")
    listener.join()
