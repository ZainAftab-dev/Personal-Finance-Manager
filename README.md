<div align="center">

# FinanceIQ — AI Financial Intelligence Platform

A personal finance dashboard that turns raw bank transactions into a real financial picture — spending trends, fraud detection, subscription tracking, budget planning, and a conversational AI assistant, all in one Streamlit app.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![Anthropic Claude](https://img.shields.io/badge/Claude_API-D97757?logo=anthropic&logoColor=white)](https://www.anthropic.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

FinanceIQ ingests bank statements (CSV, Excel, or PDF — including NayaPay exports) or synthetic demo data and turns them into a full financial intelligence dashboard: an isolation-forest fraud detector, automatic subscription/recurring-payment detection, a composite 0–100 financial health score, spending forecasts, and an AI chat assistant that answers questions about your own transaction data in plain English.

It started as a coursework project (`app.py` / `ai_utils.py`) and was extended into a fuller product (`app_enhanced.py` / `ai_utils_enhanced.py`) — the original files are kept untouched alongside the enhanced version so both are runnable.

## Screenshots

| | |
|---|---|
| **Dashboard** | **AI Chat Assistant** |
| ![Dashboard](screenshots/01_dashboard.png) | ![AI Chat Assistant](screenshots/02_ai_chat_assistant.png) |
| **Analytics** | **Fraud Detection** |
| ![Analytics](screenshots/03_analytics.png) | ![Fraud Detection](screenshots/04_fraud_detection.png) |
| **Recurring Payments** | **Financial Health** |
| ![Recurring Payments](screenshots/05_recurring_payments.png) | ![Financial Health](screenshots/06_financial_health.png) |
| **Smart Alerts** | **Budget Planning** |
| ![Smart Alerts](screenshots/07_smart_alerts.png) | ![Budget Planning](screenshots/08_budget_planning.png) |
| **Monthly Report** | **Bill Reminders** |
| ![Monthly Report](screenshots/09_monthly_report.png) | ![Bill Reminders](screenshots/10_bill_reminders.png) |
| **OCR Receipt Upload** | **Reports & Export** |
| ![OCR Receipt Upload](screenshots/11_ocr_receipt_upload.png) | ![Reports & Export](screenshots/12_reports_export.png) |

## Features

**Core Analytics**
- Overview dashboard: net-balance trend (week / month / year) with the period's peak marked, spending by category against a budget, top subscriptions with renewal dates, and a finance-assistant panel that surfaces the most unusual charge and a computed savings tip
- Detailed analytics — category, merchant, day-of-week, and income-vs-expense trend views
- Monthly reports with a month picker, category charts, and full transaction listings

**Machine Learning**
- Fraud detection via `IsolationForest`, with an anomaly scatter plot and severity scoring
- Automatic recurring-payment / subscription detection with billing-cycle inference
- Linear-trend spending forecasts and a 5-dimension financial health score (savings ratio, consistency, subscription load, risk profile, income stability)
- Statistical anomaly detection (Z-score) as a secondary fraud signal

**AI Assistant**
- Conversational chat over your own transaction data, with a provider fallback chain: Claude API → free HuggingFace Inference API → rule-based offline mode (never a dead end, even with no API key)
- Claude Vision (or local Tesseract) OCR for receipt/screenshot uploads, auto-extracting merchant, amount, and date
- PDF bank statement parsing, including a NayaPay-specific parser

**Planning & Tools**
- Budget planning against industry-standard allocation percentages
- Bill reminders derived from detected recurring charges
- Smart alerts engine (spending spikes, fraud, underused subscriptions, low savings rate)
- Excel/CSV export for monthly summaries, annual summaries, category analysis, and subscription audits
- A synthetic data generator for demoing the app without real financial data

## Tech Stack

| Layer | Tools |
|---|---|
| App framework | Streamlit |
| Data | Pandas, NumPy |
| ML | scikit-learn (`IsolationForest`, `KMeans`, `LinearRegression`), SciPy |
| Visualization | Plotly |
| AI / LLM | Anthropic Claude API, HuggingFace Inference API |
| Document parsing | pdfplumber, pytesseract, Pillow |
| Export | openpyxl |

## Getting Started

### Prerequisites
- Python 3.10+
- (Optional) [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) if you want local OCR instead of Claude Vision

### Installation

```bash
git clone https://github.com/ZainAftab-dev/Personal-Finance-Manager.git
cd Personal-Finance-Manager
pip install -r requirements_enhanced.txt
```

### Configure API keys (optional)

The app runs fully offline with rule-based fallbacks if you skip this — AI chat, OCR, and forecasting all degrade gracefully. To enable the AI features, create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=sk-ant-...        # Claude chat + Claude Vision OCR
HUGGINGFACE_API_KEY=hf_...          # free fallback for AI chat — https://huggingface.co/settings/tokens
GEMINI_API_KEY=                     # optional, rarely needed
```

### Run

```bash
streamlit run app_enhanced.py
```

The original, unmodified coursework version is still runnable independently:

```bash
streamlit run app.py
```

## Project Structure

```
├── app_enhanced.py          # Main app — UI, pages, data loading
├── ai_utils_enhanced.py     # AI chat, OCR, PDF parsing, health score, alerts
├── app.py                   # Original coursework version (preserved, unmodified)
├── ai_utils.py               # Original ML utilities (preserved, unmodified)
├── requirements_enhanced.txt # Full dependency list for app_enhanced.py
├── requirements.txt          # Minimal dependency list for app.py
├── sample_transactions.csv   # Bundled sample dataset
├── test_transactions.csv     # Extra test dataset
├── create_excel.py           # Builds a sample Excel statement for upload testing
├── generate_test_excel.py    # Generates test Excel files
├── screenshots/              # Images used in this README
├── UPGRADE_README.md         # What changed between the original and enhanced versions
└── .env                      # API keys (not committed)
```

## Data Privacy

All processing happens locally in your own Streamlit session — transaction data is never sent anywhere except to the AI provider you've explicitly configured a key for (and only for the specific chat message or receipt image you submit).

## License

MIT — see [LICENSE](LICENSE).

## Team

BS Software Engineering, COMSATS University Islamabad (Lahore Campus).

| Name | Registration No. |
|---|---|
| Zain Aftab | FA24-BSE-155 |
| Hunain Riasat | FA24-BSE-083 |
| Shaheer Aftab | FA24-BSE-089 |

Maintained by **Zain Aftab** · [GitHub](https://github.com/ZainAftab-dev) · [LinkedIn](https://www.linkedin.com/in/zain-aftab-563727295/)
