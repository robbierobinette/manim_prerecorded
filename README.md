## manim-voiceover-recorded

A RecordedService for manim-voiceover that enables the use of pre-recorded audio files with automatic transcript matching.

## Installation

```bash
pip install manim-voiceover-recorded


## Usage
```python
from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover_recorded import RecordedService

class MyScene(VoiceoverScene):
    def construct(self):
        self.set_speech_service(
            RecordedService(
                prerecorded_audio="path/to/your/audio.mp3",
                transcription_model="base"
            )
        )
        
        with self.voiceover(text="This is my text") as tracker:
            self.play(Create(Circle()))
:0

## Features
# manim-voiceover-recorded

A RecordedService for manim-voiceover that enables the use of pre-recorded audio files with automatic transcript matching.

## Installation

```bash
pip install manim-voiceover-recorded
```

## Usage

### Basic Usage

```python
from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover_recorded import RecordedService

class MyScene(VoiceoverScene):
    def construct(self):
        self.set_speech_service(
            RecordedService(
                prerecorded_audio="path/to/your/audio.mp3",
                transcription_model="base"
            )
        )
        
        with self.voiceover(text="This is my text") as tracker:
            self.play(Create(Circle()))
```

### Advanced Configuration

```python
self.set_speech_service(
    RecordedService(
        prerecorded_audio="recording.mp3",
        transcription_model="base",        # Whisper model size
        similarity_threshold=0.6,          # Minimum match score (0-1)
        global_speed=1.0,                 # Playback speed adjustment
        cache_dir="./voiceovers",         # Custom cache directory
    )
)
```

### Handling Multiple Segments

The service automatically tracks position in the audio file, so you can use multiple voiceover blocks:

```python
class MultiSegmentScene(VoiceoverScene):
    def construct(self):
        self.set_speech_service(
            RecordedService(prerecorded_audio="full_recording.mp3")
        )
        
        # First segment
        with self.voiceover(text="Welcome to this demonstration") as tracker:
            self.play(FadeIn(Text("Demo")))
            
        # Second segment - automatically continues from where first left off
        with self.voiceover(text="Let's explore some shapes") as tracker:
            self.play(Create(Square()))
```

## Features

### 🎯 Intelligent Text Matching
- **Fuzzy matching** with configurable similarity threshold
- **Phonetic similarity detection** for common transcription errors (e.g., "night" matches "nite")
- **Dynamic programming alignment** using modified Needleman-Wunsch algorithm
- **Automatic segment extraction** from longer recordings

### 🎙️ Audio Processing
- **Automatic transcription** using OpenAI Whisper
- **Word-level timing precision** for accurate synchronization
- **Support for MP3 and WAV** formats
- **FFmpeg-based audio extraction** for efficient segment processing

### 🚀 Performance & Reliability
- **Intelligent caching** to avoid reprocessing
- **Sequential matching** - tracks position to prevent reusing segments
- **Configurable playback speed** with automatic timing adjustment
- **Robust error handling** with helpful error messages

### 🔧 Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prerecorded_audio` | str | required | Path to the audio file |
| `transcription_model` | str | "base" | Whisper model size (tiny/base/small/medium/large) |
| `similarity_threshold` | float | 0.6 | Minimum match score (0.0-1.0) |
| `global_speed` | float | 1.0 | Playback speed multiplier |
| `cache_dir` | str | "./voiceovers" | Directory for cached files |

## Requirements

- Python 3.7+
- manim-voiceover
- ffmpeg (system installation)
- fuzzywuzzy[speedup]
- OpenAI Whisper (installed with manim-voiceover[transcribe])

## Common Use Cases

### 1. Pre-recorded Lectures
Record your entire lecture audio first, then sync animations:

```python
self.set_speech_service(
    RecordedService(
        prerecorded_audio="lecture_recording.mp3",
        similarity_threshold=0.7  # Higher threshold for prepared content
    )
)
```

### 2. Podcast/Interview Integration
Extract specific segments from longer recordings:

```python
self.set_speech_service(
    RecordedService(
        prerecorded_audio="interview.mp3",
        transcription_model="medium"  # Better accuracy for multiple speakers
    )
)
```

### 3. Multi-language Content
Use recordings in any language supported by Whisper:

```python
self.set_speech_service(
    RecordedService(
        prerecorded_audio="spanish_narration.mp3",
        transcription_model="large"  # Best for non-English
    )
)
```

## Troubleshooting

### No Good Match Found
If you see "No suitable match found" errors:
1. Lower the `similarity_threshold` (e.g., to 0.5)
2. Check that your text roughly matches the audio content
3. Use a larger Whisper model for better transcription

### FFmpeg Not Found
Install FFmpeg:
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Performance Issues
- Use smaller Whisper models (tiny/base) for faster processing
- Enable caching by reusing the same `cache_dir`
- Pre-transcribe long files and adjust `similarity_threshold`

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built on top of [manim-voiceover](https://github.com/ManimCommunity/manim-voiceover)
- Uses [OpenAI Whisper](https://github.com/openai/whisper) for transcription
- Inspired by the need for flexible voiceover workflows in educational content