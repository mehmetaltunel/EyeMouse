from pynput import keyboard
print(f"Listener: {keyboard.Listener}")
print(f"Key: {keyboard.Key}")
try:
    print(f"Cmd: {keyboard.Key.cmd}")
except:
    print("Cmd missing")
