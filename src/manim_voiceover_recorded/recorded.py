from pathlib import Path
import os
from manim_voiceover.services.base import SpeechService
from manim_voiceover.helper import remove_bookmarks, prompt_ask_missing_extras
from manim import logger
import json
import re

try:
    from fuzzywuzzy import fuzz
except ImportError:
    logger.warning("fuzzywuzzy not installed. Install with 'pip install fuzzywuzzy[speedup]' for better performance.")
    fuzz = None


class RecordedService(SpeechService):
    """Speech service that uses pre-recorded audio files with transcript matching."""

    def __init__(
        self,
        prerecorded_audio: str,
        transcription_model: str = "base",
        similarity_threshold: float = 0.6,
        end_buffer: float = 0.1,
        **kwargs,
    ):
        """Initialize the speech service.

        Args:
            prerecorded_audio (str): Path to the pre-recorded audio file (wav or mp3).
            transcription_model (str, optional): The OpenAI Whisper model to use for transcription. Defaults to "base".
            similarity_threshold (float, optional): Minimum similarity score for text matching (0.0-1.0). Defaults to 0.6.
        """
        # Check if the audio file exists
        if not os.path.exists(prerecorded_audio):
            raise FileNotFoundError(f"Pre-recorded audio file not found: {prerecorded_audio}")
        
        self.prerecorded_audio = prerecorded_audio
        self.similarity_threshold = similarity_threshold
        self._full_transcription = None
        self._word_boundaries = []  # Flattened list of all words with timestamps
        self._last_word_timestamp = 0.0  # Track timestamp of last word used
        self._end_buffer = end_buffer
        
        # Initialize parent class with transcription model
        super().__init__(transcription_model=transcription_model, **kwargs)
        
        # Transcribe the full audio file once
        self._transcribe_full_audio()

    def _transcribe_full_audio(self):
        """Transcribe the full pre-recorded audio file."""
        if self._whisper_model is None:
            raise RuntimeError("Whisper model not available. Make sure transcription is properly set up.")
        
        logger.info(f"Transcribing full audio file: {self.prerecorded_audio}")
        
        # Use whisper to transcribe the full audio
        transcription_result = self._whisper_model.transcribe(
            self.prerecorded_audio, **self.transcription_kwargs
        )
        
        # Store the full transcription
        self._full_transcription = transcription_result.text
        segments = transcription_result.segments

        # Flatten segments into word boundaries
        self._word_boundaries = []
        for segment in segments:
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    w = word.word.lower()
                    w = w.replace('-', ' ')
                    w = re.sub(r'[^\w\s]', '', w)
                    w = w.strip()
                    self._word_boundaries.append({
                        'word': w,
                        'start': word.start,
                        'end': word.end
                    })
        
        logger.info(f"Full transcription completed: {len(segments)} segments found, {len(self._word_boundaries)} words extracted")
        logger.debug(f"Full transcript: {self._full_transcription}")

    def _find_best_match(self, target_text: str):
        """Find the best matching word sequence for the target text."""
        # Convert target text to word list
        target_text = re.sub(r'-', ' ', target_text)
        target_text = re.sub(r'[^\w\s]', '', target_text)
        target_text = re.sub(r'\s+', ' ', target_text)
        target_words = target_text.lower().split()
        
        # Get script words starting from last timestamp
        start_idx = 0
        for i, word_data in enumerate(self._word_boundaries):
            if word_data['start'] >= self._last_word_timestamp:
                start_idx = i
                break
        
        # Extract words from remaining script
        script_words = [w['word'] for w in self._word_boundaries[start_idx:]]
        
        if not target_words or not script_words:
            return 0.0, [], ""
        
        # Find best match using the superior algorithm
        relative_start, relative_end, score = self._find_best_match_indices(target_words, script_words)
        
        if relative_start == -1:
            return score, [], ""
        
        # Convert relative indices to absolute indices
        absolute_start = start_idx + relative_start
        absolute_end = start_idx + relative_end
        word_indices = list(range(absolute_start, absolute_end + 1))
        
        # Get matched text
        matched_text = " ".join([self._word_boundaries[i]['word'] for i in word_indices])
        
        return score, word_indices, matched_text
    
    def _find_best_match_indices(self, phrase: list, script: list) -> tuple:
        """
        Find the best match for a phrase within a script using sequence alignment techniques.
        
        Args:
            phrase: List of words to search for
            script: List of words in the full script
            
        Returns:
            tuple: (start_index, end_index, score) where score is between 0 and 1
        """
        if not phrase or not script:
            return -1, -1, 0.0
        
        if len(phrase) > len(script):
            return -1, -1, 0.0
        
        best_start = -1
        best_end = -1
        best_score = 0.0
        
        # Sliding window approach
        for i in range(len(script) - len(phrase) + 1):
            # Try different window sizes to account for insertions/deletions
            if phrase[0] == script[i]:
                score = self._calculate_alignment_score(phrase, script[i:i + window_size])
                if score > best_score:
                    best_score = score
                    best_start = i
                    best_end = i + window_size - 1
        
        return best_start, best_end, best_score
    
    def _calculate_alignment_score(self, phrase: list, window: list) -> float:
        """
        Calculate alignment score between phrase and window using dynamic programming.
        Similar to Needleman-Wunsch algorithm but adapted for words.
        """
        m, n = len(phrase), len(window)
        
        # Initialize scoring matrix
        dp = [[0.0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Gap penalty
        gap_penalty = -0.5
        
        # Initialize first row and column with gap penalties
        for i in range(1, m + 1):
            dp[i][0] = i * gap_penalty
        for j in range(1, n + 1):
            dp[0][j] = j * gap_penalty
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Calculate match/mismatch score
                match_score = self._word_similarity(phrase[i-1], window[j-1])
                
                # Take maximum of three possibilities
                dp[i][j] = max(
                    dp[i-1][j-1] + match_score,  # Match/mismatch
                    dp[i-1][j] + gap_penalty,     # Deletion in window
                    dp[i][j-1] + gap_penalty      # Insertion in window
                )
        
        # Calculate normalized score
        max_possible_score = max(m, n)  # Perfect match score
        raw_score = dp[m][n]
        
        # Normalize to 0-1 range
        normalized_score = (raw_score + abs(min(m, n) * gap_penalty)) / (max_possible_score + abs(min(m, n) * gap_penalty))
        return max(0.0, min(1.0, normalized_score))
    
    def _word_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate similarity between two words.
        Returns 1.0 for exact match, partial score for similar words.
        """
        word1, word2 = word1.lower(), word2.lower()
        
        if word1 == word2:
            return 1.0
        
        # Check for common transcription errors
        if self._are_phonetically_similar(word1, word2):
            return 0.8
        
        # Use fuzzywuzzy for partial matching if available
        if fuzz:
            ratio = fuzz.ratio(word1, word2) / 100.0  # fuzz.ratio returns 0-100
            return ratio * 0.5  # Scale down non-exact matches
        else:
            # Fallback to simple length-based similarity if fuzzywuzzy not available
            distance = abs(len(word1) - len(word2))
            max_len = max(len(word1), len(word2))
            if max_len == 0:
                return 1.0
            similarity = 1.0 - (distance / max_len)
            return max(0.0, similarity * 0.5)
    
    def _are_phonetically_similar(self, word1: str, word2: str) -> bool:
        """
        Check if two words are phonetically similar (common transcription errors).
        This is a simplified version - could be enhanced with proper phonetic algorithms.
        """
        # Common transcription confusion pairs
        similar_sounds = [
            ('ph', 'f'), ('ck', 'k'), ('c', 'k'), ('s', 'z'),
            ('tion', 'shun'), ('sion', 'shun'), ('ture', 'cher'),
            ('our', 'or'), ('ough', 'uff'), ('ight', 'ite'),
            ('eigh', 'ay'), ('ieu', 'ew'), ('eau', 'o')
        ]
        
        # Apply common replacements
        w1_modified = word1.lower()
        w2_modified = word2.lower()
        
        for sound1, sound2 in similar_sounds:
            w1_modified = w1_modified.replace(sound1, sound2)
            w2_modified = w2_modified.replace(sound1, sound2)
        
        return w1_modified == w2_modified

    def _extract_audio_segment(self, word_indices: list, output_path: str):
        """Extract audio segment based on word indices."""
        if not word_indices:
            raise ValueError("No words provided for extraction")
        
        # Get start and end times from word boundaries
        start_time = self._word_boundaries[word_indices[0]]['start'] 
        end_time = self._word_boundaries[word_indices[-1]]['end'] + self._end_buffer
        
        # Update last word timestamp to the end of this segment
        self._last_word_timestamp = end_time
        
        # Use ffmpeg to extract the segment
        try:
            import subprocess
            
            cmd = [
                'ffmpeg', '-i', self.prerecorded_audio,
                '-ss', str(start_time),
                '-to', str(end_time),
                '-c', 'copy',
                '-y',  # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                # If copy codec fails, try re-encoding
                cmd = [
                    'ffmpeg', '-i', self.prerecorded_audio,
                    '-ss', str(start_time),
                    '-to', str(end_time),
                    '-y',  # Overwrite output file
                    output_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg failed: {result.stderr}")
                    
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg to use RecordedService.")

    def generate_from_text(
        self, text: str, cache_dir: str = None, path: str = None, **kwargs
    ) -> dict:
        """Generate audio from text by finding matching segments in pre-recorded audio."""
        
        # Remove bookmarks
        input_text = remove_bookmarks(text)
        
        if cache_dir is None:
            cache_dir = self.cache_dir

        input_data = {
            "input_text": input_text,
            "prerecorded_audio": self.prerecorded_audio,
            "similarity_threshold": self.similarity_threshold,
            "service": "recorded",
        }
        
        # Check cache first
        cached_result = self.get_cached_result(input_data, cache_dir)
        if cached_result is not None:
            return cached_result

        # Find the best matching word sequence
        match_score, word_indices, match_text = self._find_best_match(input_text)
        
        # Check if match is good enough
        if match_score < self.similarity_threshold:
            logger.error(f"No good match found for text: '{input_text}'")
            logger.error(f"Best match (score: {match_score:.3f}): '{match_text}'")
            raise ValueError(
                f"No suitable match found for text: '{input_text}'\n"
                f"Best match (similarity: {match_score:.3f}): '{match_text}'\n"
                f"Minimum similarity threshold: {self.similarity_threshold}"
            )
        
        logger.info(f"Found match (score: {match_score:.3f}) for: '{input_text}'")
        logger.info(f"Matched text: '{match_text}'")
        
        # Determine output path
        if path is None:
            audio_path = self.get_audio_basename(input_data) + ".mp3"
        else:
            audio_path = path
        
        full_output_path = str(Path(cache_dir) / audio_path)
        
        # Extract the audio segment
        self._extract_audio_segment(word_indices, full_output_path)
        
        # Generate word boundaries for the extracted segment
        word_boundaries = []
        if word_indices:
            current_text_offset = 0
            segment_start_time = self._word_boundaries[word_indices[0]]['start']
            
            for word_idx in word_indices:
                word_data = self._word_boundaries[word_idx]
                # Adjust timing relative to extracted segment
                adjusted_start = word_data['start'] - segment_start_time
                word_boundaries.append({
                    "audio_offset": int(adjusted_start * 1000),  # Convert to milliseconds
                    "text_offset": current_text_offset,
                    "word_length": len(word_data['word']),
                    "text": word_data['word'],
                    "boundary_type": "Word",
                })
                current_text_offset += len(word_data['word'])
        
        # Create return dictionary
        json_dict = {
            "input_text": text,
            "input_data": input_data,
            "original_audio": audio_path,
            "word_boundaries": word_boundaries,
            "transcribed_text": match_text,
            "match_score": match_score,
            "matched_word_indices": word_indices,
        }

        return json_dict