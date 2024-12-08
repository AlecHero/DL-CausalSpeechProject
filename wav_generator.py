import numpy as np
import wave
import os

def save_to_wav(data, sample_rate=16000, output_filename="output.wav"):
    # Make sure data is in the expected shape
    if len(data.shape) != 3 or data.shape[0] != 1 or data.shape[1] != 1:
        print(data.shape)
        raise ValueError("Input data must have shape [1, 1, length]")
    
    # Extract the sequence of amplitudes
    amplitudes = data[0, 0, :]
    
    # Ensure the amplitudes are in the range of int16 (-32768 to 32767)
    amplitudes = np.clip(amplitudes, -1.0, 1.0)  # Clipping to valid range if values are normalized
    int_amplitudes = (amplitudes * 32767).astype(np.int16)  # Convert to int16

    # Create the WAV file
    with wave.open(output_filename, "w") as wav_file:
        # Set the parameters for the WAV file
        n_channels = 1  # Mono audio
        sampwidth = 2  # 2 bytes for 16-bit audio
        n_frames = len(int_amplitudes)
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        
        # Write the data to the WAV file
        wav_file.writeframes(int_amplitudes.tobytes())

    print(f"WAV file saved to: {os.path.abspath(output_filename)}")

# Example usage:
if __name__ == "__main__":
    # Create dummy data in the shape [1, 1, length]
    length = 16000  # Example length
    sample_rate = 16000  # Example sample rate (16kHz)
    data = np.random.uniform(-1, 1, (1, 1, length)).astype(np.float32)

    # Save to a WAV file
    save_to_wav(data, sample_rate)
