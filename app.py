from flask import Flask, request, jsonify
import base64
import io
import numpy as np
import librosa
from scipy import stats

app = Flask(__name__)

# -----------------------------
# Helper: Compute statistics
# -----------------------------
def compute_stats(arr):
    arr = np.array(arr)

    # Handle empty case safely
    if len(arr) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "variance": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "mode": 0.0,
            "range": 0.0
        }

    mode_val = stats.mode(arr, keepdims=True).mode[0]

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "variance": float(np.var(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "mode": float(mode_val),
        "range": float(np.max(arr) - np.min(arr))
    }


# -----------------------------
# Main API Endpoint
# -----------------------------
@app.route('/analyze', methods=['POST'])
def analyze_audio():
    try:
        data = request.get_json()

        # Validate input
        if not data or "audio_base64" not in data:
            return jsonify({"error": "Invalid input"}), 400

        audio_base64 = data["audio_base64"]

        # -----------------------------
        # Decode base64 audio
        # -----------------------------
        audio_bytes = base64.b64decode(audio_base64)
        audio_stream = io.BytesIO(audio_bytes)

        # -----------------------------
        # Load audio using librosa
        # -----------------------------
        y, sr = librosa.load(audio_stream, sr=None)

        # Convert to numpy array
        y = np.array(y, dtype=np.float64)

        # -----------------------------
        # Compute statistics
        # -----------------------------
        stats_result = compute_stats(y)

        # -----------------------------
        # Build STRICT response
        # -----------------------------
        response = {
            "rows": int(len(y)),
            "columns": ["amplitude"],
            "mean": {"amplitude": stats_result["mean"]},
            "std": {"amplitude": stats_result["std"]},
            "variance": {"amplitude": stats_result["variance"]},
            "min": {"amplitude": stats_result["min"]},
            "max": {"amplitude": stats_result["max"]},
            "median": {"amplitude": stats_result["median"]},
            "mode": {"amplitude": stats_result["mode"]},
            "range": {"amplitude": stats_result["range"]},
            "allowed_values": {},
            "value_range": {
                "amplitude": [
                    stats_result["min"],
                    stats_result["max"]
                ]
            },
            "correlation": []
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
