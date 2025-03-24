# Transcription

## How to use
After deploying your REST API, you need to open the audio file in binary read mode, create the body and use the following route ```/v1/audio/transcriptions```:

```
import requests 


audio_path = "/path/to/audio.mp4"

URL = os.getenv('HOST')+':'+os.getenv('PORT')+'/'+os.getenv('API_ENDPOINT_PREFIX')
ENDPOINT = "/v1/audio/transcription"

# data for the request
with open(audio_path, 'rb') as f:
  data = {
    "model": "my_model",
    "temperature" : 0.0,
    "language": 'fr'
  }

  response = requests.post(URL+ENDPOINT, json=data, files={'file': f})
```

## Warning

Regarding vllm versions until 0.8.1, OpenAI Transcription API client does not support streaming.

