import pylangacq
import soundfile as sf
from pydub import AudioSegment
import json
import os
import argparse

def process_cha_audio(cha_file, audio_file, output_dir="processed_audio", type="speech_disorder"):
    """
    Processes a .cha file and its associated audio to create segmented MP3s
    and metadata JSON files, extracting participant info.

    Args:
        cha_file (str): Path to the .cha file.
        audio_file_base (str): Base name of the audio file (e.g., 'my_recording').
                             Assumes audio is in a format pydub can handle (like wav).
        output_dir (str): Directory to save the processed audio and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        chat_transcript = pylangacq.read_chat(cha_file)
    except Exception as e:
        print(f"Error parsing {cha_file}: {e}")
        print(f"Skipping file: {cha_file}")
        return

    try:
        audio = AudioSegment.from_file(audio_file) # Assuming .wav, adjust if needed

        for i, utterance in enumerate(chat_transcript.utterances()):
            try:
                speaker = utterance.participant
                if not speaker == 'CHI':
                    continue
                tokens = utterance.tokens
                text = ' '.join([token.word for token in tokens])
                
                # Check if time_marks exists and is not None before accessing
                if utterance.time_marks is None or len(utterance.time_marks) < 2:
                    continue
                    
                start_time_ms = utterance.time_marks[0]
                end_time_ms = utterance.time_marks[1]

                if start_time_ms is not None and end_time_ms is not None:
                    start_sample = int(start_time_ms)
                    end_sample = int(end_time_ms)

                    segment = audio[start_sample:end_sample]
                    
                    # Check if segment duration is greater than 4 seconds (4000 ms)
                    segment_duration_ms = len(segment)
                    if segment_duration_ms <= 4000:
                        continue
                    
                    output_audio_file = os.path.join(output_dir, f"utterance_{i}.mp3")
                    output_metadata_file = os.path.join(output_dir, f"utterance_{i}.json")
                    
                    # Skip if files already exist
                    if os.path.exists(output_audio_file) and os.path.exists(output_metadata_file):
                        continue
                    
                    segment.export(output_audio_file, format="mp3")

                    segment_metadata = {
                        "transcription": text,
                        "disorder_class": type
                    }
                    with open(output_metadata_file, 'w') as f:
                        json.dump(segment_metadata, f, indent=4)
                    
                    print(f"Saved utterance {i}: {segment_duration_ms/1000:.2f}s - {text[:50]}{'...' if len(text) > 50 else ''}")
            except Exception:
                # Skip utterance if any exception occurs
                continue
        
        # get total number of saved utterances
        total_utterances = len(os.listdir(output_dir))
        print(f"Total utterances saved: {total_utterances}")


    except FileNotFoundError as e:
        print(f"Error: Audio file '{audio_file}' not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def process_directory(input_dir, output_dir="processed_childes_data", type="speech_disorder"):
    """Processes all .cha and corresponding audio files in the input directory."""
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".cha"):
            cha_file_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            
            # First try to find MP3 file
            audio_file_path = os.path.join(input_dir, f"{base_name}.mp3")
            
            # If MP3 not found, look for WAV file
            if not os.path.exists(audio_file_path):
                wav_file_path = os.path.join(input_dir, f"{base_name}.wav")
                if os.path.exists(wav_file_path):
                    print(f"Found WAV file for {filename}, converting to MP3...")
                    # Convert WAV to MP3
                    audio = AudioSegment.from_wav(wav_file_path)
                    audio_file_path = os.path.join(input_dir, f"{base_name}.mp3")
                    audio.export(audio_file_path, format="mp3")
                    print(f"Converted {wav_file_path} to {audio_file_path}")
                else:
                    print(f"Warning: Found .cha file '{filename}' but no corresponding audio file (neither .mp3 nor .wav).")
                    continue

            print(f"Processing: {filename} and {os.path.basename(audio_file_path)}")
            base_name = os.path.splitext(os.path.basename(cha_file_path))[0]
            output_subdir = os.path.join(output_dir, base_name)
            os.makedirs(output_subdir, exist_ok=True)
            process_cha_audio(cha_file_path, audio_file_path, output_subdir, type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process CHILDES .cha files and associated audio.")
    parser.add_argument("--input_directory", help="Path to the input directory containing .cha and audio files.")
    parser.add_argument("--output_directory", help="Path to the output directory.")
    parser.add_argument("--type", help="The type of data being processed.")

    args = parser.parse_args()

    input_directory = args.input_directory
    output_directory = args.output_directory
    type = args.type

    process_directory(input_directory, output_directory, type)