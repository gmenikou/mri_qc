# ACR MRI Large Phantom QC Reporter

This Streamlit app lets you:
- upload your ACR MRI phantom `.txt` result files
- auto-parse the results
- classify each test as PASS / FAIL
- save every session with a timestamp in a history CSV
- generate a PDF report
- build trendlines for numeric tests over time

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files created
- `acr_qc_data/acr_qc_history.csv` keeps all previous sessions
- `acr_qc_data/reports/` stores generated PDFs
- `acr_qc_data/charts/` stores trend images

## Notes
- SNR and Central Frequency are tracked against baseline/trend rather than a universal hard fail threshold.
- LCD thresholds may vary slightly by site/protocol. The current rule flags 37 as a strong pass based on your example data.
- You can edit the pass/fail criteria in the parser functions if your local accreditation protocol differs.


## GitHub persistence
To keep history on Streamlit Cloud, set:
- GitHub owner
- repository name
- branch
- CSV path, for example `acr_qc_data/acr_qc_history.csv`
- a GitHub personal access token with permission to update repository contents

The app stores:
- `site_name`
- `scanner_name`
- `scanner_id`

Trendlines and history can then be filtered by scanner/system so multiple MRI systems stay separated cleanly inside one shared CSV.


## Streamlit secrets
For Streamlit Cloud, you can store the GitHub token securely in app secrets instead of typing it in the sidebar.

Add this in your Streamlit Cloud app settings under **Secrets**:

```toml
GITHUB_TOKEN = "your_github_token_here"
```

The app will:
- use `st.secrets["GITHUB_TOKEN"]` automatically if available
- otherwise fall back to the sidebar token input
