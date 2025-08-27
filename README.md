# Music Genre Classifier App

A web application that predicts the music genre of songs based on their valence and other audio characteristics. Built with Streamlit and Scikit-learn.

## Features

- **Valence-Based Classification**: Predicts music genre based primarily on valence value (musical positiveness)
- **Multiple Feature Input**: Adjust various audio characteristics to fine-tune predictions
- **Real-time Predictions**: Instant genre classification with confidence scores
- **User-Friendly Interface**: Simple sliders and input fields for easy interaction
- **Sample Presets**: Try pre-configured settings for different music genres

## How It Works

This application uses a machine learning model trained on audio features to classify songs into genres. The primary feature used is **valence** - a measure from 0.0 to 1.0 that describes the musical positiveness conveyed by a track (with 1.0 being the most positive).

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Steps
1. Download or clone this application
2. Navigate to the application directory
3. Install required dependencies:
```bash
pip install streamlit scikit-learn pandas numpy
```

## Usage

1. **Run the application**:
```bash
streamlit run app.py
```

2. **Open your web browser** and navigate to the local URL shown in the terminal (typically `http://localhost:8501`)

3. **Adjust the audio features** using the sliders:
   - **Valence**: The primary feature (0.0 = sad, depressed; 1.0 = happy, cheerful)
   - **Danceability**: How suitable a track is for dancing (0.0 - 1.0)
   - **Energy**: Represents intensity and activity (0.0 - 1.0)
   - **Acousticness**: Confidence measure of whether track is acoustic (0.0 - 1.0)
   - **Tempo**: The overall estimated tempo in BPM

4. **Click "Predict Genre"** to see the classification results

5. **Try preset configurations** for different genres to see how valence affects predictions

## Input Features

- **Valence (Primary)**: The musical positiveness (0.0-1.0)
- **Danceability**: How danceable the song is (0.0-1.0)
- **Energy**: Perceived intensity and activity (0.0-1.0)
- **Acousticness**: Likelihood the track is acoustic (0.0-1.0)
- **Tempo**: Speed of the song in beats per minute (50-200 BPM)

## Model Information

The classifier uses a pre-trained machine learning model (Random Forest) that was trained on thousands of songs with known genre labels and audio features.

## File Structure

```
music-classifier-app/
├── app.py                 # Main application file
├── model/                # Pre-trained model files
│   └── genre_classifier.pkl
├── requirements.txt      # Python dependencies
└── README.md
```

## Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   streamlit run app.py --server.port 8502
   ```

2. **Missing dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Model file not found**:
   - Ensure the model file is in the correct directory

## License

This project is provided for educational purposes.

## Support

For questions about this application, please check the input requirements or ensure you're using the correct valence values (0.0-1.0).
