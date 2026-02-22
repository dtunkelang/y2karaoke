import json
import sys
import os
from pathlib import Path


def create_html(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    lines = data.get("lines", [])
    title = f"{data.get('artist', 'Unknown')} - {data.get('title', 'Unknown')}"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title} - Lyrics Timeline</title>
        <style>
            body {{ font-family: sans-serif; padding: 20px; background: #f0f0f0; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px;
                          box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            h1 {{ text-align: center; color: #333; }}
            .line {{ padding: 10px; border-bottom: 1px solid #eee; display: flex; align-items: center; }}
            .line:nth-child(even) {{ background: #fafafa; }}
            .time {{ font-family: monospace; color: #666; margin-right: 15px; width: 120px; flex-shrink: 0; }}
            .text {{ font-size: 1.1em; color: #222; flex-grow: 1; }}
            .confidence {{ font-size: 0.8em; color: #aaa; width: 60px; text-align: right; }}
            .words {{ display: flex; flex-wrap: wrap; gap: 5px; margin-top: 5px; }}
            .word {{ background: #e0e7ff; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }}
            .word-time {{ font-size: 0.7em; color: #666; margin-left: 3px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            <div id="lyrics">
    """

    for i, line in enumerate(lines):
        start = line.get("start", 0)
        end = line.get("end", 0)
        text = line.get("text", "")
        conf = line.get("confidence", 0)
        words = line.get("words", [])

        words_html = ""
        if words:
            words_html = '<div class="words">'
            for w in words:
                w_text = w.get("text", "")
                w_start = w.get("start", 0)
                words_html += f'<span class="word">{w_text}<span class="word-time">({w_start:.2f})</span></span>'
            words_html += "</div>"

        html += f"""
            <div class="line">
                <div class="time">{start:.2f}s - {end:.2f}s</div>
                <div class="content">
                    <div class="text">{text}</div>
                    {words_html}
                </div>
                <div class="confidence">{conf*100:.0f}%</div>
            </div>
        """

    html += """
            </div>
        </div>
    </body>
    </html>
    """

    output_path = Path(json_path).with_suffix(".html")
    with open(output_path, "w") as f:
        f.write(html)

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_json.py <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]
    if not os.path.exists(json_file):
        print(f"File not found: {json_file}")
        sys.exit(1)

    html_file = create_html(json_file)
    print(f"Generated HTML: {html_file}")
