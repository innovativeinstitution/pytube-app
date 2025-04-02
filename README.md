<h1 align="center">PyTube-App</h1>

## Google API Setup for YouTube Upload
Before uploading to YouTube:

Go to the Google Cloud Console.

Create a project, enable the YouTube Data API v3.

Create OAuth 2.0 Client Credentials.

Download client_secrets.json (save in same directory as main.py)

## ElevenLabs API Setup
1. Create a free account at https://elevenlabs.io/
2. Generate API key and update .env below

## Setting Up the Project

### Creating the `.env` File

Create a `.env` file in the root directory of the project and add the following environment variables:

```
OPENAI_API_KEY="your-openai-api-key"
USE_ELEVENLABS=true
ELEVENLABS_API_KEY="your-elevenlabs-api-key"
```

### Running the Project

1. **Clone the repository:**

   ```sh
   git clone https://github.com/innovativeinstitution/pytube-app.git
   cd pytube-app
   ```

2. **Install dependencies:**
   If you are using `pip`, run:

   ```sh
   pip install -r requirements.txt
   ```

3. **Run app:**

   ```sh
   python.exe main.py
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
