#!/usr/bin/env python3
"""Comprehensive audio diagnostic for Windows."""
import sys
import os

print("=" * 60)
print("COMPREHENSIVE AUDIO DIAGNOSTIC")
print("=" * 60)

# Test 1: Windows built-in beep (most reliable)
print("\n1. WINDOWS BUILT-IN BEEP (winsound.Beep)")
print("-" * 40)
try:
    import winsound
    print("   Playing 440Hz beep for 1 second...")
    winsound.Beep(440, 1000)
    print("   Done! Did you hear it?")
except Exception as e:
    print(f"   ERROR: {e}")

input("\n   Press Enter to continue to next test...")

# Test 2: Windows MessageBeep
print("\n2. WINDOWS MESSAGE BEEP (winsound.MessageBeep)")
print("-" * 40)
try:
    import winsound
    print("   Playing Windows alert sound...")
    winsound.MessageBeep(winsound.MB_OK)
    import time
    time.sleep(0.5)
    print("   Done! Did you hear it?")
except Exception as e:
    print(f"   ERROR: {e}")

input("\n   Press Enter to continue to next test...")

# Test 3: Play a WAV file
print("\n3. WINDOWS WAV FILE PLAYBACK (winsound.PlaySound)")
print("-" * 40)
try:
    import winsound
    # Play Windows default sound
    print("   Playing Windows 'SystemAsterisk' sound...")
    winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
    print("   Done! Did you hear it?")
except Exception as e:
    print(f"   ERROR: {e}")

input("\n   Press Enter to continue to next test...")

# Test 4: sounddevice with explicit settings
print("\n4. SOUNDDEVICE WITH EXPLICIT SETTINGS")
print("-" * 40)
try:
    import sounddevice as sd
    import numpy as np

    # List all output devices
    print("   Available OUTPUT devices:")
    devices = sd.query_devices()
    output_devs = []
    for i, d in enumerate(devices):
        if d['max_output_channels'] > 0:
            output_devs.append(i)
            marker = " <-- DEFAULT" if i == sd.default.device[1] else ""
            print(f"     [{i}] {d['name']}{marker}")

    # Get default
    default_out = sd.default.device[1]
    device_info = sd.query_devices(default_out)
    print(f"\n   Using device [{default_out}]: {device_info['name']}")
    print(f"   Default sample rate: {device_info['default_samplerate']}")

    # Use device's native sample rate
    sample_rate = int(device_info['default_samplerate'])
    duration = 2.0
    frequency = 440

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t) * 0.5
    tone = tone.astype(np.float32)

    print(f"\n   Playing {frequency}Hz tone at {sample_rate}Hz sample rate...")
    print(f"   Audio shape: {tone.shape}, dtype: {tone.dtype}")
    print(f"   Min: {tone.min():.3f}, Max: {tone.max():.3f}")

    sd.play(tone, sample_rate, device=default_out, blocking=True)
    print("   Done!")

except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

input("\n   Press Enter to continue to next test...")

# Test 5: Try each output device
print("\n5. TEST EACH OUTPUT DEVICE INDIVIDUALLY")
print("-" * 40)
try:
    import sounddevice as sd
    import numpy as np

    devices = sd.query_devices()

    for i, d in enumerate(devices):
        if d['max_output_channels'] > 0:
            print(f"\n   Testing device [{i}]: {d['name']}")
            try:
                sample_rate = int(d['default_samplerate'])
                duration = 0.5
                frequency = 440

                t = np.linspace(0, duration, int(sample_rate * duration), False)
                tone = np.sin(2 * np.pi * frequency * t) * 0.5
                tone = tone.astype(np.float32)

                sd.play(tone, sample_rate, device=i, blocking=True)
                print(f"     Played successfully")
            except Exception as e:
                print(f"     Failed: {e}")

            response = input("     Did you hear sound? (y/n/q to quit): ").strip().lower()
            if response == 'y':
                print(f"\n   *** WORKING DEVICE FOUND: [{i}] {d['name']} ***")
                print(f"   Add this to your .env file:")
                print(f"   AUDIO_OUTPUT_DEVICE={i}")
                break
            elif response == 'q':
                break

except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

print("""
TROUBLESHOOTING TIPS:
1. If winsound.Beep worked but nothing else did:
   - Check Windows Volume Mixer (right-click speaker icon)
   - Make sure Python.exe is not muted

2. If nothing worked:
   - Check if speakers are connected and powered on
   - Check physical volume controls
   - Try different headphones/speakers
   - Check Windows Sound settings (right-click speaker > Sound settings)

3. If only specific device worked:
   - Note the device number and add to .env:
     AUDIO_OUTPUT_DEVICE=<number>
""")
