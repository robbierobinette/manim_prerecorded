from manim import (
    Text, Circle, Square, Triangle, MathTex, Group,
    Write, Create, FadeOut, Transform,
    UP, DOWN, RIGHT,
    BLUE, RED, GREEN, YELLOW, WHITE, GRAY, ORANGE, PURPLE
)
from manim_voiceover import VoiceoverScene
from pathlib import Path
import tempfile
import os

import librosa
from manim_prerecorded.services import RecordedService

# Import for generating test audio
from gtts import gTTS


def create_test_audio(audio_path: str):
    """Create test audio file for the demo."""
    print(f"Creating test audio at: {audio_path}")
    
    # Script for our demo video
    demo_script = """
    Welcome to this demonstration of the RecordedService for Manim.
    First, we will show a simple circle animation.
    Next, we will create a square that transforms.
    Finally, we will display some text with mathematics.
    Thank you for watching this demonstration.
    """
    
    try:
        tts = gTTS(text=demo_script.strip(), lang='en', slow=False)
        tts.save(audio_path)
        print(f"✓ Test audio created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create test audio: {e}")
        return False


class RecordedServiceDemo(VoiceoverScene):
    """Demo scene using RecordedService for voiceover."""
    
    def construct(self):
        """Main scene construction."""
        
        # Set up audio file path
        audio_dir = Path("./demo_audio")
        audio_dir.mkdir(exist_ok=True)
        audio_file = audio_dir / "demo_script.mp3"
        
        # Create test audio if it doesn't exist
        if not audio_file.exists():
            print("Creating demo audio file...")
            if not create_test_audio(str(audio_file)):
                self.add(Text("Failed to create audio file", color=RED))
                return
        
        # Initialize RecordedService
        try:
            print("Initializing RecordedService...")
            recorded_service = RecordedService(
                prerecorded_audio=str(audio_file),
                transcription_model="base",
                similarity_threshold=0.3,  # Lower threshold for better matching
                cache_dir="./voiceover_cache"
            )
            
            # Set the voiceover service for this scene
            self.set_speech_service(recorded_service)
            print("✓ RecordedService initialized")
        except Exception as e:
            print(f"❌ Failed to initialize RecordedService: {e}")
            self.add(Text(f"Service Error: {str(e)}", color=RED, font_size=24))
            return
        
        # Now create the demo with voiceovers
        self.create_demo_with_voiceover()
    
    def create_demo_with_voiceover(self):
        """Create the main demo with synchronized voiceover."""
        
        # Title sequence
        title = Text("RecordedService Demo", font_size=48, color=BLUE)
        self.play(Write(title))
        
        with self.voiceover(text="Welcome to this demonstration of the RecordedService for Manim") as tracker:
            self.wait(tracker.duration)
        
        self.play(FadeOut(title))
        
        # Circle animation
        circle_title = Text("Circle Animation", font_size=36, color=GREEN)
        circle_title.to_edge(UP)
        self.play(Write(circle_title))
        
        with self.voiceover(text="First, we will show a simple circle animation") as tracker:
            circle = Circle(radius=2, color=YELLOW)
            self.play(Create(circle), run_time=tracker.duration)
        
        # Make the circle bounce
        self.play(
            circle.animate.scale(1.5).set_color(RED),
            run_time=1
        )
        self.play(
            circle.animate.scale(0.7).set_color(BLUE),
            run_time=1
        )
        
        self.play(FadeOut(circle), FadeOut(circle_title))
        
        # Square transformation
        square_title = Text("Square Transformation", font_size=36, color=PURPLE)
        square_title.to_edge(UP)
        self.play(Write(square_title))
        
        with self.voiceover(text="Next, we will create a square that transforms") as tracker:
            square = Square(side_length=2, color=ORANGE)
            self.play(Create(square), run_time=tracker.duration/2)
            
            # Transform to different shapes
            triangle = Triangle(color=GREEN)
            self.play(Transform(square, triangle), run_time=tracker.duration/2)
        
        self.play(FadeOut(square), FadeOut(square_title))
        
        # Mathematics display
        math_title = Text("Mathematics", font_size=36, color=YELLOW)
        math_title.to_edge(UP)
        self.play(Write(math_title))
        
        with self.voiceover(text="Finally, we will display some text with mathematics") as tracker:
            # Create mathematical expression
            equation = MathTex(
                r"E = mc^2",
                font_size=72,
                color=WHITE
            )
            
            formula_text = Text(
                "Einstein's famous equation",
                font_size=24,
                color=GRAY
            )
            formula_text.next_to(equation, DOWN, buff=0.5)
            
            self.play(Write(equation), run_time=tracker.duration/2)
            self.play(Write(formula_text), run_time=tracker.duration/2)
        
        self.wait(1)
        self.play(FadeOut(equation), FadeOut(formula_text), FadeOut(math_title))
        
        # Closing
        closing_text = Text("Demo Complete!", font_size=48, color=GREEN)
        self.play(Write(closing_text))
        
        with self.voiceover(text="Thank you for watching this demonstration") as tracker:
            self.wait(tracker.duration)
        
        self.play(FadeOut(closing_text))


class SimpleRecordedDemo(VoiceoverScene):
    """Simpler demo version with minimal setup."""
    
    def construct(self):
        """Simple demo that just shows basic functionality."""
        
        # Create a very simple audio file for testing
        audio_file = Path("simple_demo.mp3")
        
        if not audio_file.exists():
            simple_text = "Hello world. This is a test. Welcome to Manim."
            try:
                tts = gTTS(text=simple_text, lang='en')
                tts.save(str(audio_file))
                print(f"Created simple audio: {audio_file}")
            except Exception as e:
                print(f"Failed to create audio: {e}")
                # Fallback - show error message
                error_text = Text("Could not create audio file", color=RED)
                self.add(error_text)
                return
        
        # Initialize service
        try:
            voice_service = RecordedService(
                prerecorded_audio=str(audio_file),
                transcription_model="base",
                similarity_threshold=0.2
            )
            # Set the voiceover service for this scene
            self.set_speech_service(voice_service)
        except Exception as e:
            print(f"Service error: {e}")
            error_text = Text(f"Service Error: {str(e)}", color=RED, font_size=24)
            self.add(error_text)
            return
        
        # Simple animation with voiceover
        title = Text("Simple Demo", font_size=48)
        self.add(title)
        
        try:
            with self.voiceover(text="Hello world") as tracker:
                circle = Circle(color=BLUE)
                self.play(Create(circle), run_time=tracker.duration)
            
            with self.voiceover(text="This is a test") as tracker:
                square = Square(color=RED).next_to(circle, RIGHT)
                self.play(Create(square), run_time=tracker.duration)
            
            with self.voiceover(text="Welcome to Manim") as tracker:
                group = Group(circle, square)
                self.play(group.animate.scale(1.5), run_time=tracker.duration)
                
        except Exception as e:
            print(f"Voiceover error: {e}")
            error_text = Text(f"Voiceover Error: {str(e)}", color=RED, font_size=18)
            error_text.next_to(title, DOWN)
            self.add(error_text)


def main():
    """Main function to provide usage instructions."""
    print("RecordedService Manim Demo")
    print("="*50)
    print("\nThis file contains two demo scenes:")
    print("1. RecordedServiceDemo - Full featured demo")
    print("2. SimpleRecordedDemo - Minimal demo for testing")
    print("\nTo run:")
    print("manim manim_recorded_demo.py RecordedServiceDemo")
    print("or")
    print("manim manim_recorded_demo.py SimpleRecordedDemo")
    print("\nMake sure you have:")
    print("- RecordedService in manim_voiceover/services/recorded.py")
    print("- Dependencies: pip install gtts stable-whisper librosa soundfile")


if __name__ == "__main__":
    main()