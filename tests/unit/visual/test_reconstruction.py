from y2karaoke.core.refine_visual import reconstruct_lyrics_from_visuals
from y2karaoke.core.models import TargetLine

def test_reconstruct_lyrics_single_frame_single_line():
    raw_frames = [
        {
            "time": 1.0,
            "words": [
                {"text": "Hello", "x": 10, "y": 100, "w": 50, "h": 20},
                {"text": "World", "x": 70, "y": 100, "w": 50, "h": 20},
            ]
        }
    ]
    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    
    # Should detect 1 line
    assert len(lines) == 1
    assert lines[0].text == "Hello World"
    # Start time snapped from 1.0
    assert lines[0].start == 1.0
    # End time snapped from last seen (1.0) + 2.0 = 3.0
    assert lines[0].end == 3.0

def test_reconstruct_lyrics_detects_multiple_lines():
    raw_frames = [
        {
            "time": 1.0,
            "words": [
                {"text": "Line", "x": 10, "y": 100, "w": 50, "h": 20},
                {"text": "One", "x": 70, "y": 100, "w": 50, "h": 20},
                {"text": "Line", "x": 10, "y": 200, "w": 50, "h": 20},
                {"text": "Two", "x": 70, "y": 200, "w": 50, "h": 20},
            ]
        }
    ]
    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    
    assert len(lines) == 2
    assert lines[0].text == "Line One"
    assert lines[1].text == "Line Two"

def test_reconstruct_lyrics_tracks_duration():
    # Line appears at 1.0, disappears after 2.0
    raw_frames = [
        {
            "time": 1.0,
            "words": [{"text": "Hello", "x": 10, "y": 100, "w": 50, "h": 20}]
        },
        {
            "time": 2.0,
            "words": [{"text": "Hello", "x": 10, "y": 100, "w": 50, "h": 20}]
        },
        {
            "time": 3.0,
            "words": [] # Disappeared
        },
        # Ensure it's committed (needs > 1.0s gap)
        {
            "time": 4.5,
            "words": []
        }
    ]
    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    
    assert len(lines) == 1
    assert lines[0].start == 1.0
    # Last seen 2.0. End = 2.0 + 2.0 = 4.0
    assert lines[0].end == 4.0

def test_reconstruct_lyrics_deduplicates_similar_lines():
    # Line appears, disappears, reappears slightly shifted (noise/jitter)
    raw_frames = [
        {
            "time": 1.0,
            "words": [{"text": "Hello", "x": 10, "y": 100, "w": 50, "h": 20}]
        },
        {
            "time": 3.0,
            "words": [] # Gap > 1.0s
        },
        {
            "time": 3.5,
            "words": [{"text": "Hello", "x": 12, "y": 102, "w": 50, "h": 20}] # Shifted
        },
        {
            "time": 5.0,
            "words": [] # Commit
        }
    ]
    lines = reconstruct_lyrics_from_visuals(raw_frames, 1.0)
    
    # Should probably merge or deduplicate if they are considered "same occurrence"
    # But current logic only deduplicates if they overlap in time/space?
    # Logic: "if abs(ent["y"] - u["y"]) < 20 and abs(ent["first"] - u["first"]) < 2.0"
    # Here first=1.0 and first=3.5. Diff = 2.5. Not deduplicated by default logic.
    
    # Wait, if they are temporally distinct, they SHOULD stay distinct (repeated line).
    assert len(lines) == 2
    
    # Try one that SHOULD be deduplicated (close in time/space)
    raw_frames_dup = [
        {
            "time": 1.0,
            "words": [{"text": "Hello", "x": 10, "y": 100, "w": 50, "h": 20}]
        },
        {
            "time": 2.0, # Gap < 1.0s? No, just later frame.
            "words": [] 
        },
        {
            "time": 2.5, # first=2.5. Diff=1.5. < 2.0.
            "words": [{"text": "Hello", "x": 10, "y": 100, "w": 50, "h": 20}]
        },
        {
            "time": 4.0,
            "words": []
        }
    ]
    # This might fail if the first line is committed before the second appears?
    # Logic: "Commit lines that have disappeared ... if frame["time"] - on_screen[nt]["last"] > 1.0"
    # At 2.0, last=1.0. Diff=1.0. Not > 1.0.
    # At 2.5, "Hello" reappears. It matches `norm` key "y3_hello".
    # So it updates `last`.
    
    # If key matches, it's the SAME line object.
    
    lines_dup = reconstruct_lyrics_from_visuals(raw_frames_dup, 1.0)
    assert len(lines_dup) == 1
    assert lines_dup[0].start == 1.0
    # Last seen 2.5. End = 4.5.
    assert lines_dup[0].end == 4.5
