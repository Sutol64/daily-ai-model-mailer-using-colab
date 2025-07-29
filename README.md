# Daily AI Model Mailer using LoRA and Colab

This project uses a Hugging Face-hosted LoRA model (`Woman877.v2`) to generate a daily image and send it via email using Google Colab.

## Setup

1. Fill in the secrets directly in the notebook:
   - `HF_TOKEN`: Hugging Face read token.
   - `GMAIL_USER`: Gmail address.
   - `GMAIL_PASS`: Gmail App password (not account password).
   - `TO_EMAIL`: Where to send the image.

2. Open the notebook in Colab and run all cells.

3. Optionally set up a scheduler using [colab.research.google.com](https://colab.research.google.com) + [GitHub Actions External Trigger](https://github.com/features/actions).
