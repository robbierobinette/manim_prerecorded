from pathlib import Path
import os
import json
import hashlib
from difflib import SequenceMatcher
import re
import time
from typing import List, Tuple, Optional, Dict, Any

from manim_voiceover.services.base import SpeechService
from manim_voiceover.helper import remove_bookmarks
from manim import logger

try:
    import whisper
    import stable_whisper
    from pydub import AudioSegment
    import numpy as np
    import fuzzy
except ImportError as e:
    logger.error(
        'Missing packages. Run `pip install "manim-voiceover[transcribe]" pydub fuzzy` to use RecordedService.'
    )
    raise ImportError(
        'Missing required packages for RecordedService. '
        'Please install with: pip install "manim-voiceover[transcribe]" pydub fuzzy'
    ) from e


class RecordedService(SpeechService):
    """Speech service that uses pre-recorded audio files with intelligent text matching."""

    def __init__(
        self,
        prerecorded_audio: str,
        transcription_model: str = "base",
        similarity_threshold: float = 0.6,
        **kwargs,
    ):
        """Initialize the speech service.

        Args:
            prerecorded_audio (str): Path to the pre-recorded audio file (wav or mp3).
            transcription_model (str, optional): The OpenAI Whisper model to use for transcription. Defaults to "base".
            similarity_threshold (float, optional): Minimum similarity score for text matching. Defaults to 0.6.
        """
        super().__init__(transcription_model=transcription_model, **kwargs)
        
        self.prerecorded_audio = Path(prerecorded_audio)
        self.similarity_threshold = similarity_threshold
        
        if not self.prerecorded_audio.exists():
            raise FileNotFoundError(f"Pre-recorded audio file not found: {prerecorded_audio}")
        
        # Set up transcription cache
        self.transcription_cache_file = Path(self.cache_dir) / f"{self.prerecorded_audio.stem}_transcription.json"
        
        # Initialize DMetaphone object
        self.dmeta = fuzzy.DMetaphone()
        
        # Load and transcribe the pre-recorded audio (with caching)
        self._load_and_transcribe()

    def _load_and_transcribe(self):
        """Load the pre-recorded audio and generate transcript with word-level timestamps."""
        
        # Check if we have a cached transcription
        audio_mtime = self.prerecorded_audio.stat().st_mtime
        
        if self.transcription_cache_file.exists():
            try:
                with open(self.transcription_cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache is still valid (audio file hasn't changed)
                if cache_data.get('audio_mtime') == audio_mtime:
                    logger.info(f"Using cached transcription for: {self.prerecorded_audio}")
                    self.full_transcript = cache_data['full_transcript']
                    self.word_segments = cache_data['word_segments']
                    
                    # Ensure dmetaphone codes are tuples (in case they were loaded from JSON as lists)
                    for word_info in self.word_segments:
                        if isinstance(word_info['dmetaphone'], list):
                            word_info['dmetaphone'] = tuple(word_info['dmetaphone'])
                    
                    # Log the loaded transcription
                    logger.info(f"Loaded cached transcription. Found {len(self.word_segments)} words.")
                    logger.info("Complete transcription with timing:")
                    for i, word_info in enumerate(self.word_segments):
                        logger.info(f"  [{i:3d}] {word_info['start']:6.2f}s - {word_info['end']:6.2f}s: '{word_info['word']}'")
                    return
                else:
                    logger.info("Audio file has changed, re-transcribing...")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Invalid transcription cache, re-transcribing: {e}")
        
        # Transcribe the audio
        logger.info(f"Transcribing pre-recorded audio: {self.prerecorded_audio}")
        
        # Use stable-whisper for word-level timestamps
        if self._whisper_model is None:
            self._whisper_model = stable_whisper.load_model(self.transcription_model)
        
        self.transcription_result = self._whisper_model.transcribe(
            str(self.prerecorded_audio), 
            suppress_silence=False,
            **self.transcription_kwargs
        )
        
        self.full_transcript = self.transcription_result.text
        self.word_segments = []
        
        # Extract word-level information
        for segment in self.transcription_result.segments:
            for word_info in segment.words:
                raw_word = word_info.word.strip()
                # Normalize transcript words once: lowercase and remove punctuation
                normalized_word = re.sub(r'[^\w\s]', '', raw_word.lower()).strip()
                # Create dmetaphone codes for matching (cached once here)
                dmetaphone_codes = self.dmeta(normalized_word)
                # Convert bytes to strings for JSON serialization
                dmetaphone_codes_str = tuple(
                    code.decode('utf-8') if isinstance(code, bytes) else code 
                    for code in dmetaphone_codes
                )
                
                self.word_segments.append({
                    'word': normalized_word,
                    'dmetaphone': dmetaphone_codes_str,
                    'start': word_info.start,
                    'end': word_info.end
                })
        
        # Log the complete transcription with timing
        logger.info(f"Transcription complete. Found {len(self.word_segments)} words.")
        logger.info("Complete transcription with timing:")
        for i, word_info in enumerate(self.word_segments):
            logger.info(f"  [{i:3d}] {word_info['start']:6.2f}s - {word_info['end']:6.2f}s: '{word_info['word']}'")
        
        # Cache the transcription
        cache_data = {
            'audio_mtime': audio_mtime,
            'full_transcript': self.full_transcript,
            'word_segments': self.word_segments,
            'transcription_model': self.transcription_model
        }
        
        with open(self.transcription_cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger.info(f"Transcription cached to: {self.transcription_cache_file}")

    def _convert_dmetaphone_codes(self, codes: tuple) -> tuple:
        """Convert dmetaphone codes to strings for consistency."""
        return tuple(
            code.decode('utf-8') if isinstance(code, bytes) else code 
            for code in codes
        )

    def _dmetaphone_codes_match(self, codes1: tuple, codes2: tuple) -> bool:
        """Check if two dmetaphone code tuples match.
        
        DMetaphone returns a tuple of (primary, secondary) codes.
        Match if any non-None code matches between the tuples.
        """
        # Filter out None values and compare
        valid_codes1 = [c for c in codes1 if c is not None]
        valid_codes2 = [c for c in codes2 if c is not None]
        
        # Check if any code from set 1 matches any code from set 2
        return any(c1 == c2 for c1 in valid_codes1 for c2 in valid_codes2)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison: handle hyphens, remove punctuation, convert to lowercase."""
        # Step 1: Replace hyphens with spaces first (before removing other punctuation)
        text = text.replace('-', ' ')
        # Step 2: Convert to lowercase
        text = text.lower()
        # Step 3: Remove all punctuation except spaces
        text = re.sub(r'[^\w\s]', '', text)
        # Step 4: Normalize whitespace (remove extra spaces)
        return ' '.join(text.split())

    def _needleman_wunsch_alignment(self, target_codes: List[tuple], script_codes: List[tuple]) -> List[Tuple[Optional[int], Optional[int]]]:
        """
        Perform Needleman-Wunsch sequence alignment to find optimal alignment between two sequences.
        Now uses dmetaphone codes instead of direct string comparison.
        
        Args:
            target_codes: First sequence (target text dmetaphone codes)
            script_codes: Second sequence (transcript dmetaphone codes)
            
        Returns:
            List of tuples representing alignment: (index_in_seq1, index_in_seq2)
            None values indicate gaps in the alignment.
        """
        len1, len2 = len(target_codes), len(script_codes)
        
        # Initialize scoring matrix
        # Match: +2, Mismatch: -1, Gap: -1
        score_matrix = np.zeros((len1 + 1, len2 + 1), dtype=int)
        
        # Initialize gap penalties
        for i in range(len1 + 1):
            score_matrix[i][0] = -i
        for j in range(len2 + 1):
            score_matrix[0][j] = -j
        
        # Fill the scoring matrix using dmetaphone matching
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                # Use dmetaphone matching instead of direct comparison
                match = self._dmetaphone_codes_match(target_codes[i-1], script_codes[j-1])
                match_score = score_matrix[i-1][j-1] + (2 if match else -1)
                gap_seq1 = score_matrix[i-1][j] - 1
                gap_seq2 = score_matrix[i][j-1] - 1
                
                score_matrix[i][j] = max(match_score, gap_seq1, gap_seq2)
        
        # Traceback to find alignment
        alignment = []
        i, j = len1, len2
        
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                match = self._dmetaphone_codes_match(target_codes[i-1], script_codes[j-1])
                if score_matrix[i][j] == score_matrix[i-1][j-1] + (2 if match else -1):
                    alignment.append((i-1, j-1))
                    i -= 1
                    j -= 1
                elif score_matrix[i][j] == score_matrix[i-1][j] - 1:
                    alignment.append((i-1, None))
                    i -= 1
                else:
                    alignment.append((None, j-1))
                    j -= 1
            elif i > 0:
                alignment.append((i-1, None))
                i -= 1
            else:
                alignment.append((None, j-1))
                j -= 1
        
        alignment.reverse()
        return alignment

    def _calculate_alignment_score(self, target_codes: List[tuple], script_segment_codes: List[tuple]) -> float:

        """
        Calculate alignment score between phrase and script segment using Needleman-Wunsch.
        Now properly penalizes segments with extra mismatched words.
        
        Args:
            target_codes: Target phrase dmetaphone codes
            script_segment_codes: Segment of script dmetaphone codes to compare against
            
        Returns:
            Alignment score (0.0 to 1.0)
        """
        if not target_codes or not script_segment_codes:
            return 0.0
        
        # Perform sequence alignment with dmetaphone codes
        alignment = self._needleman_wunsch_alignment(target_codes, script_segment_codes)
        
        # Calculate alignment statistics
        matches = 0
        target_gaps = 0  # Gaps in target (extra words in script)
        script_gaps = 0  # Gaps in script (missing words from target)
        
        for t_idx, s_idx in alignment:
            if t_idx is not None and s_idx is not None:
                # Both sequences have a word at this position
                if self._dmetaphone_codes_match(target_codes[t_idx], script_segment_codes[s_idx]):
                    matches += 1
            elif t_idx is None:
                # Gap in target sequence (extra word in script)
                target_gaps += 1
            else:
                # Gap in script sequence (missing word from target)
                script_gaps += 1
        
        # Calculate penalties
        total_target_words = len(target_codes)
        total_script_words = len(script_segment_codes)
        
        # Base score: proportion of target words that matched
        base_score = matches / total_target_words
        
        # Penalty for extra words in script segment
        # The more extra words, the lower the score
        length_penalty = 1.0 - (target_gaps / max(total_script_words, 1))
        
        # Penalty for missing words from target
        completeness_penalty = 1.0 - (script_gaps / max(total_target_words, 1))
        
        # Combine scores with weights
        # Base score is most important, then completeness, then length
        final_score = (
            base_score * 0.6 +           # 60% weight on actual matches
            completeness_penalty * 0.3 + # 30% weight on completeness
            length_penalty * 0.1         # 10% weight on avoiding extra words
        )
        
        return min(final_score, 1.0)  # Cap at 1.0


    def find_best_match_indices(self, target_words: List[str]) -> Tuple[int, int, float]:
        """
        Find the best matching indices for a phrase within the script.
        Now with improved scoring that penalizes segments with extra mismatched words.
        
        Args:
            target_words: List of normalized words from target text
            
        Returns:
            Tuple of (start_index, end_index, best_score)
        """
        if not target_words:
            raise ValueError("Target words cannot be empty")
        
        # Create dmetaphone codes for target words
        target_codes = [self._convert_dmetaphone_codes(self.dmeta(word)) for word in target_words]
        
        first_word_codes = target_codes[0]
        last_word_codes = target_codes[-1]
        best_score = 0
        best_start = 0
        best_end = len(target_words)
        
        # Find all positions where the first word matches using dmetaphone
        first_word_positions = [
            i for i, word_info in enumerate(self.word_segments) 
            if self._dmetaphone_codes_match(first_word_codes, word_info['dmetaphone'])
        ]
        
        if not first_word_positions:
            raise ValueError(f"First word '{target_words[0]}' not found in script")
        
        # Find all positions where the last word matches using dmetaphone
        last_word_positions = [
            i for i, word_info in enumerate(self.word_segments) 
            if self._dmetaphone_codes_match(last_word_codes, word_info['dmetaphone'])
        ]
        
        if not last_word_positions:
            raise ValueError(f"Last word '{target_words[-1]}' not found in script")
        
        # Try all combinations of start and end positions
        for start_idx in first_word_positions:
            for end_idx in last_word_positions:
                # End index must be after start index
                if end_idx <= start_idx:
                    continue
                
                # Don't search segments that are too long (more than 2x the phrase length)
                # Reduced from 3x to 2x to be more restrictive
                segment_length = end_idx - start_idx + 1
                if segment_length > len(target_words) * 2:
                    continue
                
                # Extract script segment dmetaphone codes
                script_segment_codes = [
                    self.word_segments[i]['dmetaphone'] 
                    for i in range(start_idx, end_idx + 1)
                ]
                
                score = self._calculate_alignment_score(target_codes, script_segment_codes)
                
                # Log for debugging
                matched_words = [self.word_segments[i]['word'] for i in range(start_idx, end_idx + 1)]
                print(f"start_idx: {start_idx}, end_idx: {end_idx}, score: {score:.3f}, segment: '{' '.join(matched_words)}'")
                
                if score > best_score:
                    best_score = score
                    best_start = start_idx
                    best_end = end_idx + 1  # +1 because end_idx should be exclusive for slicing
        
        return best_start, best_end, best_score 

    def _find_best_match(self, target_text: str) -> Tuple[int, int, float]:
        """
        Find the best matching segment in the transcript using dmetaphone-based alignment.
        
        Args:
            target_text: The text to find in the transcript
            
        Returns:
            Tuple of (start_word_index, end_word_index, similarity_score)
        """
        target_words = self._normalize_text(target_text).split()
        
        if not target_words:
            raise ValueError("Target text is empty after normalization")
        
        # Use the helper method to find best match
        return self.find_best_match_indices(target_words)

    def _extract_audio_segment(self, start_word_idx: int, end_word_idx: int, output_path: str):
        """Extract audio segment based on word indices."""
        if start_word_idx >= len(self.word_segments) or end_word_idx > len(self.word_segments):
            raise IndexError("Word indices out of range")
        
        start_time = self.word_segments[start_word_idx]['start']
        end_time = self.word_segments[end_word_idx - 1]['end']
        
        # Log the timing information
        logger.info(f"Extracting audio segment:")
        logger.info(f"  Start word [{start_word_idx}]: '{self.word_segments[start_word_idx]['word']}' at {start_time:.2f}s")
        logger.info(f"  End word [{end_word_idx-1}]: '{self.word_segments[end_word_idx-1]['word']}' at {end_time:.2f}s")
        logger.info(f"  Segment duration: {end_time - start_time:.2f}s")
        logger.info(f"  Output file: {output_path}")
        
        # Load audio and extract segment
        audio = AudioSegment.from_file(str(self.prerecorded_audio))
        
        # Convert to milliseconds
        start_ms = max(int(start_time * 1000) - 100, 0)
        end_ms = int(end_time * 1000) + 300
        
        logger.info(f"FFmpeg timing: start={start_ms}ms, end={end_ms}ms")
        
        # Extract segment
        segment = audio[start_ms:end_ms]
        
        # Export as MP3
        segment.export(output_path, format="mp3")
        
        return start_time, end_time

    def generate_from_text(
        self, text: str, cache_dir: str = None, path: str = None, **kwargs
    ) -> dict:
        """Generate audio from text by finding the best match in pre-recorded audio."""
        
        # Remove bookmarks
        input_text = remove_bookmarks(text)
        
        if cache_dir is None:
            cache_dir = self.cache_dir

        input_data = {
            "input_text": input_text,
            "prerecorded_audio": str(self.prerecorded_audio),
            "similarity_threshold": self.similarity_threshold,
            "service": "recorded",
        }
        
        # Check cache
        cached_result = self.get_cached_result(input_data, cache_dir)
        if cached_result is not None:
            return cached_result

        # Find best match
        try:
            start_idx, end_idx, similarity = self._find_best_match(input_text)
        except Exception as e:
            logger.error(f"Error finding match for text: {input_text}")
            raise e

        if similarity < self.similarity_threshold:
            # Extract the matched segment for comparison
            matched_words = [self.word_segments[i]['word'] for i in range(start_idx, end_idx)]
            matched_text = ' '.join(matched_words)
            
            error_msg = f"""
Text matching failed (similarity: {similarity:.3f} < {self.similarity_threshold}):

Target text: "{input_text}"
Best match:  "{matched_text}"

Please adjust the text or lower the similarity_threshold.
            """
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Generate output filename
        if path is None:
            audio_path = self.get_audio_basename(input_data) + ".mp3"
        else:
            audio_path = path

        output_path = Path(cache_dir) / audio_path
        
        # Extract audio segment
        start_time, end_time = self._extract_audio_segment(start_idx, end_idx, str(output_path))
        
        # Generate word boundaries for the extracted segment
        word_boundaries = []
        current_text_offset = 0
        
        for i in range(start_idx, end_idx):
            word_info = self.word_segments[i]
            word = word_info['word']
            
            # Adjust timing relative to segment start
            adjusted_start = word_info['start'] - start_time
            
            word_boundaries.append({
                "audio_offset": int(adjusted_start * 1000),  # Convert to milliseconds
                "text_offset": current_text_offset,
                "word_length": len(word),
                "text": word,
                "boundary_type": "Word",
            })
            current_text_offset += len(word)

        # Prepare return dictionary
        json_dict = {
            "input_text": text,
            "input_data": input_data,
            "original_audio": audio_path,
            "word_boundaries": word_boundaries,
            "transcribed_text": ' '.join([self.word_segments[i]['word'] for i in range(start_idx, end_idx)]),
            "similarity_score": similarity,
            "matched_segment": {
                "start_time": start_time,
                "end_time": end_time,
                "start_word_idx": start_idx,
                "end_word_idx": end_idx,
            }
        }

        logger.info(f"Successfully matched text (similarity: {similarity:.3f})")
        logger.info(f"Extracted audio segment: {start_time:.2f}s - {end_time:.2f}s")
        
        return json_dict