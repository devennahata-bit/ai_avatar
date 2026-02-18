#!/usr/bin/env python3
"""Simple audio test - plays on system default."""
import numpy as np

print("Simple Audio Test")
print("=" * 40)

# Try sounddevice
try:
    import sounddevice as sd

    print(f"\nDefault devices: {sd.default.device}")
    print(f"Default output: {sd.query_devices(sd.default.device[1])['name']}")

    # Generate tone
    duration = 2.0
    sample_rate = 44100
    frequency = 440

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t) * 0.5
    tone = tone.astype(np.float32)

    print(f"\nPlaying {frequency}Hz tone for {duration}s on DEFAULT device...")
    print("(If you don't hear this, check Windows volume mixer)")

    sd.play(tone, sample_rate)  # No device specified = use default
    sd.wait()

    print("Done!")

except Exception as e:
    print(f"sounddevice error: {e}")

    # Fallback: try winsound (Windows built-in)
    print("\nTrying Windows built-in beep...")
    try:
        import winsound
        winsound.Beep(440, 1000)  # 440Hz for 1 second
        print("winsound beep done!")
    except Exception as e2:
        print(f"winsound error: {e2}")

# Also try pygame as fallback
print("\nTrying pygame mixer...")
try:
    import pygame
    pygame.mixer.init(frequency=44100)

    # Generate tone as pygame sound
    duration = 1.0
    sample_rate = 44100
    frequency = 440

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t) * 0.3 * 32767
    tone = tone.astype(np.int16)

    # Stereo
    stereo = np.column_stack((tone, tone))
    sound = pygame.sndarray.make_sound(stereo)

    print("Playing via pygame...")
    sound.play()
    pygame.time.wait(1500)
    pygame.mixer.quit()
    print("pygame done!")
except ImportError:
    print("pygame not installed (optional)")
except Exception as e:
    print(f"pygame error: {e}")

print("\n" + "=" * 40)
print("If you heard nothing, check:")
print("1. Windows volume (not muted)")
print("2. Correct output device in Windows Sound settings")
print("3. Speaker/headphone connection")
