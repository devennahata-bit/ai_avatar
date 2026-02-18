#!/usr/bin/env python3
"""Test audio playback and configure devices."""

import os
import queue
import subprocess
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv


def _safe_input(prompt: str) -> str:
    """Read user input safely; return empty string in non-interactive runs."""
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def _resolve_choice(choice: str, devices: list[tuple[int, dict]], default_index: int) -> int:
    """Map a 1-based menu choice to an audio device index."""
    if not choice:
        return default_index
    try:
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(devices):
            return devices[choice_idx][0]
    except ValueError:
        pass
    return default_index


def main() -> int:
    load_dotenv()

    print("=" * 60)
    print("AUDIO DEVICE TEST AND CONFIGURATION")
    print("=" * 60)

    # Install sounddevice if needed
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        print("Installing sounddevice...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sounddevice"])
        import sounddevice as sd
        import numpy as np

    # Show all audio devices
    print("\n1. AVAILABLE AUDIO DEVICES:")
    print("-" * 60)
    devices = sd.query_devices()

    input_devices: list[tuple[int, dict]] = []
    output_devices: list[tuple[int, dict]] = []

    for i, device in enumerate(devices):
        device_type = []
        if device["max_input_channels"] > 0:
            device_type.append("INPUT")
            input_devices.append((i, device))
        if device["max_output_channels"] > 0:
            device_type.append("OUTPUT")
            output_devices.append((i, device))
        type_str = "/".join(device_type) if device_type else "NONE"

        name = device["name"]
        marker = ""
        if "realtek" in name.lower():
            marker = " *** REALTEK ***"
        elif "speakers" in name.lower() or "headphone" in name.lower():
            marker = " <-- SPEAKERS"
        elif "microphone" in name.lower() or "mic" in name.lower():
            marker = " <-- MIC"

        print(f"  [{i:2d}] {name}{marker}")
        print(f"       Type: {type_str}, In: {device['max_input_channels']}, Out: {device['max_output_channels']}")

    # Show current defaults
    print("\n2. CURRENT DEFAULT DEVICES:")
    print("-" * 60)
    default_input, default_output = sd.default.device
    if default_input is not None:
        print(f"  Default INPUT:  [{default_input}] {sd.query_devices(default_input)['name']}")
    else:
        print("  Default INPUT:  None")
    if default_output is not None:
        print(f"  Default OUTPUT: [{default_output}] {sd.query_devices(default_output)['name']}")
    else:
        print("  Default OUTPUT: None")

    print("\n3. SELECT DEVICES:")
    print("-" * 60)
    print("\n  Available OUTPUT devices (speakers):")
    for i, (idx, dev) in enumerate(output_devices):
        print(f"    {i + 1}. [{idx}] {dev['name']}")

    output_choice = _safe_input("\n  Enter OUTPUT device number (or press Enter for default): ")
    selected_output = _resolve_choice(output_choice, output_devices, default_output)
    print(f"  Selected OUTPUT: [{selected_output}] {sd.query_devices(selected_output)['name']}")

    print("\n  Available INPUT devices (microphone):")
    for i, (idx, dev) in enumerate(input_devices):
        print(f"    {i + 1}. [{idx}] {dev['name']}")

    input_choice = _safe_input("\n  Enter INPUT device number (or press Enter for default): ")
    selected_input = _resolve_choice(input_choice, input_devices, default_input)
    print(f"  Selected INPUT: [{selected_input}] {sd.query_devices(selected_input)['name']}")

    # Test audio output
    print("\n4. TESTING AUDIO OUTPUT (playing test tone)...")
    print("-" * 60)
    try:
        duration = 1.0
        frequency = 440
        sample_rate = 24000

        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * frequency * t) * 0.3
        tone = tone.astype(np.float32)

        print(f"  Playing 440Hz tone on device [{selected_output}]...")
        sd.play(tone, sample_rate, device=selected_output)
        sd.wait()
        print("  Done! Did you hear the beep?")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Trying with default device...")
        try:
            sd.play(tone, sample_rate)
            sd.wait()
            print("  Played on default device.")
        except Exception as e2:
            print(f"  ERROR with default: {e2}")

    # Test microphone
    print("\n5. TESTING MICROPHONE (recording 2 seconds)...")
    print("-" * 60)
    try:
        print(f"  Recording from device [{selected_input}] for 2 seconds...")
        print("  Speak now!")

        recording = sd.rec(
            int(2 * 16000),
            samplerate=16000,
            channels=1,
            dtype="float32",
            device=selected_input,
        )
        sd.wait()

        max_level = np.max(np.abs(recording))
        print(f"  Recording complete. Max audio level: {max_level:.4f}")
        if max_level > 0.01:
            print("  Microphone is working!")
        else:
            print("  WARNING: Very low audio level. Check microphone.")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Trying with default device...")
        try:
            recording = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype="float32")
            sd.wait()
            max_level = np.max(np.abs(recording))
            print(f"  Recording complete. Max audio level: {max_level:.4f}")
        except Exception as e2:
            print(f"  ERROR with default: {e2}")

    # Save device config
    print("\n6. SAVING DEVICE CONFIGURATION...")
    print("-" * 60)
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                env_content = f.read()
        else:
            env_content = ""

        if "AUDIO_INPUT_DEVICE" not in env_content:
            with open(env_path, "a", encoding="utf-8") as f:
                f.write("\n# Audio Device Settings\n")
                f.write(f"AUDIO_INPUT_DEVICE={selected_input}\n")
                f.write(f"AUDIO_OUTPUT_DEVICE={selected_output}\n")
            print(f"  Added AUDIO_INPUT_DEVICE={selected_input} to .env")
            print(f"  Added AUDIO_OUTPUT_DEVICE={selected_output} to .env")
        else:
            print("  Audio device settings already in .env")
            print("  Edit .env manually to change devices.")
    except Exception as e:
        print(f"  Could not update .env: {e}")

    # Test ElevenLabs TTS
    print("\n7. TESTING ELEVENLABS TTS...")
    print("-" * 60)
    try:
        from modules.voice import VoiceSynthesizer

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            print("  ERROR: ELEVENLABS_API_KEY not set in .env")
        else:
            audio_queue = queue.Queue()
            synthesizer = VoiceSynthesizer(
                api_key=api_key,
                voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
                audio_queue=audio_queue,
                output_device=selected_output,
            )

            print("  Synthesizing and playing 'Hello, this is Aria speaking'...")
            synthesizer.synthesize_to_queue("Hello, this is Aria speaking.", play_audio=True)
            print("  TTS playback completed!")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nSelected devices:")
    print(f"  INPUT:  [{selected_input}] {sd.query_devices(selected_input)['name']}")
    print(f"  OUTPUT: [{selected_output}] {sd.query_devices(selected_output)['name']}")
    print("\nIf audio worked, run: python main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
