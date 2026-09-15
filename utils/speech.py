"""
Text-to-speech helper for Rupee Reader.

Uses the browser's built-in Web Speech API (SpeechSynthesis) instead of a
Python TTS library. This avoids platform-specific audio dependencies
(no espeak/nsss/sapi5 setup needed) and works reliably inside Streamlit,
which renders in a browser anyway.
"""

DENOMINATION_PHRASES = {
    "10": "Ten rupees",
    "20": "Twenty rupees",
    "50": "Fifty rupees",
    "100": "One hundred rupees",
    "200": "Two hundred rupees",
    "500": "Five hundred rupees",
    "2000": "Two thousand rupees",
}


def phrase_for(denomination: str) -> str:
    """Return the spoken phrase for a denomination label, e.g. '500' -> 'Five hundred rupees'."""
    return DENOMINATION_PHRASES.get(str(denomination), f"{denomination} rupees")


def speak_html(text: str) -> str:
    """Return a small HTML/JS snippet that speaks `text` aloud via the
    browser's SpeechSynthesis API when rendered (e.g. with
    st.components.v1.html)."""
    safe_text = text.replace("\\", "\\\\").replace('"', '\\"')
    return f"""
    <script>
    const msg = new SpeechSynthesisUtterance("{safe_text}");
    msg.rate = 0.95;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(msg);
    </script>
    """
