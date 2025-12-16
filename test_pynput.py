from pynput import keyboard
try:
    print(keyboard.GlobalHotkeys)
    print("GlobalHotkeys exists")
except AttributeError:
    print("GlobalHotkeys misses")
