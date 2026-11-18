# Kuku Promo Prototype

A Streamlit client prototype for finding and reviewing promo segments. A redacted 50-episode sample is loaded on first open, with a featured suggestion ready to review.

## Run locally

Install `requirements.txt`, then run `python -m streamlit run jagan.py`.

The sample hides screenplay dialogue and full transcripts. Upload a script folder in the app to analyze your own material. Model API settings are optional; local rules work without a key. Configure secrets only in the hosting service, never in this repository.

## Deploy

Deploy `jagan.py` from this repository with Streamlit Community Cloud.
