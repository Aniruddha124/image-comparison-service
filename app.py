from flask import Flask, request, jsonify
from matcher_improved import get_image_paths_from_db
from matcher_improved import find_top_matches_improved
import os

app = Flask(__name__)

@app.route('/compare', methods=['POST'])
def compare_images():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    uploaded_image = request.files['image']
    print(f"Received image: {uploaded_image.filename}")
    uploaded_path = os.path.join('temp', uploaded_image.filename)

    try:
        # Save the uploaded image to temporary location
        uploaded_image.save(uploaded_path)

        # Get comparison images from PostgreSQL via ir_attachment table
        comparison_paths = get_image_paths_from_db()

        if not comparison_paths:
            return jsonify({'error': 'No images found in database for comparison.'}), 500

        # Call matcher to get top matches
        top_matches = find_top_matches_improved(uploaded_path, comparison_paths)

        return jsonify({
            'top_matches': top_matches
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        pass
        # Clean up the uploaded image
        if os.path.exists(uploaded_path):
            os.remove(uploaded_path)

if __name__ == '__main__':
    os.makedirs("temp", exist_ok=True)
    app.run(host='0.0.0.0', port=5001, debug=True)
