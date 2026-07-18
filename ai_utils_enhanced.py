"""
AI Financial Intelligence Platform - Enhanced AI Utilities
Supports: Claude API chatbot, ML analytics, fraud detection, forecasting
Preserves all original ai_utils.py functionality + adds Claude chatbot
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')
import os
import json
import re
from collections import Counter

try:
    import streamlit as st
    _cache_data = st.cache_data(show_spinner=False)
except ImportError:
    # keep this module importable/testable outside a Streamlit context
    def _cache_data(fn): return fn

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy import stats

# ─────────────────────────────────────────────
#  CLAUDE API CHATBOT
# ─────────────────────────────────────────────

CLAUDE_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
HF_API_KEY       = os.environ.get("HUGGINGFACE_API_KEY", "")   # free tier at huggingface.co
CHATBOT_ENABLED  = bool(CLAUDE_API_KEY or HF_API_KEY)          # works with either key


def _build_financial_context(df: pd.DataFrame) -> str:
    """Build a compact financial summary to inject into the chatbot context."""
    if df is None or len(df) == 0:
        return "No transaction data loaded."

    expenses  = df[df['transaction_type'] == 'expense']
    income_df = df[df['transaction_type'] == 'income']

    total_exp = expenses['amount'].sum()
    total_inc = income_df['amount'].sum()
    balance   = total_inc - total_exp
    savings_r = (balance / total_inc * 100) if total_inc > 0 else 0

    top_cats = (expenses.groupby('category')['amount']
                .sum().sort_values(ascending=False).head(5))

    top_merch = (expenses.groupby('merchant_name')['amount']
                 .sum().sort_values(ascending=False).head(5))

    monthly = (expenses.groupby(expenses['date'].dt.to_period('M'))['amount']
               .sum().tail(3))

    ctx = f"""
FINANCIAL SUMMARY (User's Data):
- Total Income:   ${total_inc:,.2f}
- Total Expenses: ${total_exp:,.2f}
- Net Balance:    ${balance:,.2f}
- Savings Rate:   {savings_r:.1f}%
- Transactions:   {len(df)}
- Date Range:     {df['date'].min().date()} → {df['date'].max().date()}

TOP SPENDING CATEGORIES:
{chr(10).join(f'  {cat}: ${amt:,.2f}' for cat, amt in top_cats.items())}

TOP MERCHANTS:
{chr(10).join(f'  {m}: ${a:,.2f}' for m, a in top_merch.items())}

RECENT MONTHLY SPENDING:
{chr(10).join(f'  {p}: ${a:,.2f}' for p, a in monthly.items())}
""".strip()
    return ctx


def chat_with_claude(
    user_message: str,
    conversation_history: List[Dict],
    df: Optional[pd.DataFrame] = None,
) -> str:
    """
    Send a message to AI with financial context.
    Priority: Claude API → HuggingFace (free) → Gemini → rule-based
    """
    if CLAUDE_API_KEY:
        return _chat_with_claude_api(user_message, conversation_history, df)
    if HF_API_KEY:
        return _chat_with_huggingface(user_message, conversation_history, df)
    if GEMINI_API_KEY:
        return _chat_with_gemini(user_message, conversation_history, df)
    return _rule_based_chat(user_message, df)


def _chat_with_claude_api(user_message, conversation_history, df):
    """Claude API implementation."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        fin_context = _build_financial_context(df) if df is not None else ""
        system_prompt = f"""You are an expert AI Financial Analyst for a Pakistani personal finance app.
You help users understand spending habits, detect subscriptions, predict expenses, and improve financial health.
Be concise, insightful, and actionable. Use PKR for amounts. Use bullet points for lists.

{fin_context}

Rules:
- Answer only finance-related questions using the data above
- Keep responses under 200 words unless a breakdown is needed
- Do not use emoji or decorative symbols anywhere in the response
"""
        messages = []
        for turn in conversation_history[-10:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=512,
            system=system_prompt, messages=messages,
        )
        return response.content[0].text
    except Exception as e:
        print(f"[Claude API error] {e}")
        return _rule_based_chat(user_message, df)


def _chat_with_huggingface(user_message: str, history: List[Dict], df) -> str:
    """
    FREE HuggingFace Inference API — uses Mistral-7B-Instruct (completely free).
    Get key at: https://huggingface.co/settings/tokens (free account)
    """
    try:
        import requests
        fin_context = _build_financial_context(df) if df is not None else ""
        system = (f"You are an expert AI Financial Analyst for a Pakistani personal finance app. "
                  f"Be concise and use PKR for amounts.\n\n{fin_context}")
        # Build conversation string for Mistral instruct format
        prompt = f"<s>[INST] {system}\n\nUser question: {user_message} [/INST]"

        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 300, "temperature": 0.7, "return_full_text": False}
        }
        # Try Mistral first, fall back to Zephyr
        for model in ["mistralai/Mistral-7B-Instruct-v0.2", "HuggingFaceH4/zephyr-7b-beta"]:
            url = f"https://api-inference.huggingface.co/models/{model}"
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    text = data[0].get("generated_text", "")
                    if text:
                        return text.strip()
            elif resp.status_code == 503:
                # Model loading — try rule-based
                continue

        return _rule_based_chat(user_message, df)
    except Exception as e:
        print(f"[HuggingFace API error] {e}")
        return _rule_based_chat(user_message, df)


def _chat_with_gemini(user_message: str, history: List[Dict], df) -> str:
    """Gemini fallback chatbot."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-pro")
        ctx = _build_financial_context(df) if df is not None else ""
        prompt = f"You are a financial assistant.\n\nContext:\n{ctx}\n\nUser: {user_message}"
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return _rule_based_chat(user_message, df)


def _rule_based_chat(user_message: str, df) -> str:
    """Local rule-based fallback when no API keys are set."""
    if df is None or len(df) == 0:
        return "Please upload your transaction data first so I can analyse it for you."

    msg = user_message.lower()
    expenses  = df[df['transaction_type'] == 'expense']
    income_df = df[df['transaction_type'] == 'income']
    total_exp = expenses['amount'].sum()
    total_inc = income_df['amount'].sum()

    if any(w in msg for w in ['spend', 'spent', 'expense', 'cost']):
        top = expenses.groupby('category')['amount'].sum().sort_values(ascending=False).head(3)
        lines = [f"- {c}: ${a:,.2f}" for c, a in top.items()]
        return f"Your total spending is **${total_exp:,.2f}**.\n\nTop categories:\n" + "\n".join(lines)

    if any(w in msg for w in ['subscription', 'recurring', 'membership']):
        from ai_utils_enhanced import detect_recurring_payments_enhanced
        rec = detect_recurring_payments_enhanced(df)
        if len(rec) > 0:
            monthly = rec['avg_amount'].sum()
            names = ", ".join(rec['merchant_name'].head(5).tolist())
            return f"Detected **{len(rec)} subscriptions** costing ~${monthly:,.2f}/month.\n\nTop: {names}"
        return "No clear recurring subscriptions detected in your data."

    if any(w in msg for w in ['income', 'earn', 'salary', 'revenue']):
        return f"Your total income is **${total_inc:,.2f}** with a net balance of **${total_inc - total_exp:,.2f}**."

    if any(w in msg for w in ['save', 'saving', 'budget']):
        rate = ((total_inc - total_exp) / total_inc * 100) if total_inc > 0 else 0
        return (f"Savings rate: **{rate:.1f}%**. "
                + ("You're on track — aim for 20%+ for long-term wealth." if rate >= 15
                   else "Consider reducing discretionary spending to improve your savings rate."))

    if any(w in msg for w in ['suspicious', 'fraud', 'unusual', 'anomaly']):
        return "Use the **Fraud Detection** page for detailed anomaly analysis powered by Isolation Forest."

    if any(w in msg for w in ['predict', 'forecast', 'next month']):
        fc = predict_monthly_spending(df, months_ahead=2)
        if 'error' not in fc:
            items = [f"- {k}: ${v:,.0f}" for k, v in list(fc.get('forecasts', {}).items())[:2]]
            return "Spending forecast:\n" + "\n".join(items)
        return "Not enough data for forecasting (need 3+ months)."

    return (f"I'm your financial assistant. I can see **{len(df)} transactions** totalling "
            f"**${total_exp:,.2f}** in expenses. Try asking: 'How much did I spend?', "
            "'What are my subscriptions?', or 'How can I save more?'")


# ─────────────────────────────────────────────
#  PDF STATEMENT PARSER — NayaPay + generic
# ─────────────────────────────────────────────

NAYAPAY_TXN_TYPES = [
    'Peer to Peer','IBFT In','IBFT Out','Raast In','Raast Out',
    'Online','Cash In','Cash Out','Request In','Request Out','Salary','Bill Payment',
]
INCOME_TYPES = {'IBFT In','Raast In','Cash In','Request In','Salary'}


def _parse_amount_str(s: str) -> Optional[float]:
    """Convert 'Rs. 63,887.13' or '-Rs. 310' or '+Rs. 65,000' to float."""
    s = re.sub(r'Rs\.?\s*', '', str(s)).replace(',', '').replace('+', '').strip()
    try:
        return float(s)
    except Exception:
        return None


def parse_nayapay_pdf_text(full_text: str) -> pd.DataFrame:
    """
    Parse NayaPay PDF statement text.
    NayaPay format: each transaction is on one line starting with 'DD Mon YYYY  TYPE  DESCRIPTION  AMOUNT  BALANCE'
    """
    lines = [l.strip() for l in full_text.split('\n')]
    # Regex: date + rest of line containing at least one Rs. amount
    line_pat = re.compile(
        r'^(\d{2} \w{3} \d{4})\s+'   # date
        r'(.+?)\s+'                    # type + description (greedy minimal)
        r'([+-]?Rs\.?\s*[\d,]+\.?\d*)'  # amount
        r'\s+(Rs\.?\s*[\d,]+\.?\d*)'    # balance
        r'\s*$'
    )
    records = []
    for line in lines:
        m = line_pat.match(line)
        if not m:
            continue
        date_str, middle, amt_str, _bal = m.groups()
        # Identify transaction type from beginning of middle
        txn_type = 'Unknown'
        description = middle.strip()
        for t in NAYAPAY_TXN_TYPES:
            if middle.startswith(t):
                txn_type = t
                description = middle[len(t):].strip()
                break
        amount = _parse_amount_str(amt_str)
        if amount is None:
            continue
        is_income = (amount > 0 or txn_type in INCOME_TYPES)
        records.append({
            'date':             date_str,
            'merchant_name':    description.split('|')[0].split('\n')[0].strip()[:60],
            'category':         'Income' if is_income else 'Uncategorized',
            'amount':           abs(amount),
            'description':      description,
            'transaction_type': 'income' if is_income else 'expense',
        })
    return pd.DataFrame(records)


def parse_pdf_statement(file_bytes: bytes) -> Optional[pd.DataFrame]:
    """
    Extract transactions from PDF bank statements.
    Tries: (1) NayaPay text parser, (2) generic table parser.
    """
    try:
        import pdfplumber

        full_text = ''
        all_table_records = []

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ''
                full_text += page_text + '\n'
                # Also try tables
                for table in page.extract_tables():
                    if not table or len(table) < 2:
                        continue
                    headers = [str(h).lower().strip().replace(' ', '_') if h else f'c{i}'
                               for i, h in enumerate(table[0])]
                    for row in table[1:]:
                        if row and any(c for c in row if c):
                            all_table_records.append(
                                dict(zip(headers, [str(c).strip() if c else '' for c in row]))
                            )

        # Try NayaPay text parser first
        if 'NayaPay' in full_text or 'TIMESTAMP' in full_text or 'Peer to Peer' in full_text:
            df = parse_nayapay_pdf_text(full_text)
            if len(df) > 0:
                df['transaction_id'] = [f'TXN{i:06d}' for i in range(len(df))]
                return df

        # Try generic table records
        if all_table_records:
            df = pd.DataFrame(all_table_records)
            return df  # caller will normalize

        # Generic line-by-line text approach
        records = []
        date_pat   = re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{2} \w{3} \d{4}|\d{4}-\d{2}-\d{2}')
        amount_pat = re.compile(r'[+-]?(?:Rs\.?\s*)?[\d,]+\.?\d{0,2}')
        for line in full_text.split('\n'):
            line = line.strip()
            if date_pat.search(line) and amount_pat.search(line):
                ds    = date_pat.search(line).group()
                amts  = [p for p in line.split() if re.match(r'^[+-]?[\d,]+\.?\d{0,2}$', p.replace(',', ''))]
                desc  = re.sub(r'[0-9/\-\.,+Rs]+', ' ', line).strip()
                if amts:
                    records.append({'date': ds, 'description': desc, 'amount': amts[-1].replace(',', '')})

        return pd.DataFrame(records) if records else None

    except ImportError:
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────
#  OCR IMAGE PARSER
# ─────────────────────────────────────────────

def parse_image_receipt(file_bytes: bytes) -> str:
    """Extract text from receipt/screenshot using Tesseract OCR."""
    try:
        from PIL import Image
        import pytesseract
        import io as _io

        # Common Windows Tesseract install paths — try all
        for tess_path in [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\Public\Tesseract-OCR\tesseract.exe',
            '/usr/bin/tesseract', '/usr/local/bin/tesseract',
        ]:
            import os as _os
            if _os.path.exists(tess_path):
                pytesseract.pytesseract.tesseract_cmd = tess_path
                break

        img = Image.open(_io.BytesIO(file_bytes))
        # Upscale for better OCR accuracy
        w, h = img.size
        if w < 800:
            img = img.resize((w * 2, h * 2), Image.LANCZOS)
        text = pytesseract.image_to_string(img, config='--psm 6')
        return text.strip()

    except ImportError:
        return "TESSERACT_NOT_INSTALLED"
    except Exception as e:
        return f"OCR_ERROR: {e}"


def extract_transaction_from_ocr(ocr_text: str) -> Optional[Dict]:
    """
    Parse OCR text to extract transaction details.
    Works with NayaPay, JazzCash, Easypaisa receipts.
    """
    lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]
    result = {
        'merchant': None, 'amount': None, 'date': None,
        'txn_id': None, 'from': None, 'to': None, 'raw_text': ocr_text
    }

    # Amount patterns — handles Rs., PKR, plain numbers
    amount_patterns = [
        re.compile(r'Rs\.?\s*([\d,]+\.?\d{0,2})', re.IGNORECASE),
        re.compile(r'PKR\s*([\d,]+\.?\d{0,2})', re.IGNORECASE),
        re.compile(r'Amount[:\s]+(?:Rs\.?)?\s*([\d,]+\.?\d{0,2})', re.IGNORECASE),
        re.compile(r'^([\d,]+\.\d{2})\s*$'),
    ]
    date_pattern = re.compile(
        r'(\d{1,2}\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', re.IGNORECASE
    )
    tid_pattern  = re.compile(r'(?:TID|Transaction\s*ID|Ref)[:\s#]+([A-Za-z0-9]{6,})', re.IGNORECASE)
    to_pattern   = re.compile(r'(?:To|Transferred\s+to|Recipient)[:\s]+(.+)', re.IGNORECASE)
    from_pattern = re.compile(r'(?:From|Sender)[:\s]+(.+)', re.IGNORECASE)

    for line in lines:
        # Amount
        if result['amount'] is None:
            for pat in amount_patterns:
                m = pat.search(line)
                if m:
                    try:
                        result['amount'] = float(m.group(1).replace(',', ''))
                        # Set merchant from context if it's a known service
                        for svc in ['JazzCash','NayaPay','Easypaisa','HBL','Meezan','Foodpanda','Careem','Daraz']:
                            if svc.lower() in ocr_text.lower():
                                result['merchant'] = svc
                    except Exception:
                        pass
                    break

        # Date
        if result['date'] is None:
            dm = date_pattern.search(line)
            if dm:
                result['date'] = dm.group(1)

        # Transaction ID
        if result['txn_id'] is None:
            tm = tid_pattern.search(line)
            if tm:
                result['txn_id'] = tm.group(1)

        # To/From
        tom = to_pattern.search(line)
        if tom and result['to'] is None:
            result['to'] = tom.group(1).strip()[:50]
        fm = from_pattern.search(line)
        if fm and result['from'] is None:
            result['from'] = fm.group(1).strip()[:50]

    # If merchant still unknown, use first meaningful line
    if result['merchant'] is None and lines:
        for line in lines[:5]:
            if len(line) > 3 and not re.match(r'^[\d\s\-/]+$', line):
                result['merchant'] = line[:50]
                break

    return result if result['amount'] else None


# ─────────────────────────────────────────────
#  NAYAPAY CSV PARSER  (handles the header rows)
# ─────────────────────────────────────────────

def parse_nayapay_csv(file_bytes: bytes) -> Optional[pd.DataFrame]:
    """
    Parse NayaPay CSV export format.
    NayaPay CSVs have ~13 header rows before the actual TIMESTAMP/TYPE/DESCRIPTION table.
    """
    try:
        import io as _io
        raw = file_bytes.decode('utf-8', errors='replace')
        lines = raw.split('\n')

        # Find the header row (contains TIMESTAMP or TIME and TYPE)
        data_start = None
        for i, line in enumerate(lines):
            if ('TIMESTAMP' in line or 'TIME' in line) and 'TYPE' in line and 'AMOUNT' in line:
                data_start = i
                break

        if data_start is None:
            return None

        data_str = '\n'.join(lines[data_start:])
        df = pd.read_csv(_io.StringIO(data_str))

        # Standardize column names
        df.columns = [c.strip().upper() for c in df.columns]
        time_col = next((c for c in df.columns if 'TIME' in c or 'DATE' in c), None)
        amt_col  = next((c for c in df.columns if 'AMOUNT' in c), None)
        desc_col = next((c for c in df.columns if 'DESC' in c or 'NARR' in c or 'DETAIL' in c), None)
        type_col = next((c for c in df.columns if 'TYPE' in c), None)

        if time_col is None or amt_col is None:
            return None

        # Parse date
        df['date'] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=['date'])

        # Parse amount — remove Rs., commas, handle + prefix
        df['amount_raw'] = (df[amt_col].astype(str)
                            .str.replace(r'Rs\.?\s*', '', regex=True)
                            .str.replace(',', '')
                            .str.replace('+', '')
                            .str.strip())
        df['amount'] = pd.to_numeric(df['amount_raw'], errors='coerce')
        df = df.dropna(subset=['amount'])

        # Merchant name from description
        if desc_col:
            # NayaPay descriptions use | and \n as separators
            df['merchant_name'] = (df[desc_col].astype(str)
                                   .str.split('|').str[0]
                                   .str.split('\n').str[0]
                                   .str.strip()
                                   .str[:60])
        else:
            df['merchant_name'] = 'Unknown'

        # Transaction type
        income_keywords = ['IBFT In','Raast In','Cash In','Request In','Salary','IBFT IN','RAAST IN']
        if type_col:
            df['transaction_type'] = df.apply(
                lambda r: 'income' if (r['amount'] > 0 or any(k.lower() in str(r[type_col]).lower() for k in income_keywords))
                else 'expense', axis=1
            )
        else:
            df['transaction_type'] = df['amount'].apply(lambda x: 'income' if x > 0 else 'expense')

        df['amount'] = df['amount'].abs()
        df['category'] = df['merchant_name'].apply(_auto_categorize_simple)
        df['description'] = df['merchant_name']
        df['transaction_id'] = [f'TXN{i:06d}' for i in range(len(df))]

        return df[['date','merchant_name','category','amount','description','transaction_type','transaction_id']].sort_values('date').reset_index(drop=True)

    except Exception:
        return None


def _auto_categorize_simple(merchant: str) -> str:
    """Simple keyword categorizer for use inside ai_utils_enhanced."""
    m = str(merchant).lower()
    kw = {
        'Food & Dining':  ['foodpanda','kfc','mcdonald','pizza','cafe','biryani','restaurant','cheetay','careem food'],
        'Utilities':      ['lesco','gepco','ssgc','sngpl','ptcl','jazz','telenor','zong','ufone','electric','water'],
        'Transportation': ['careem','indrive','bykea','pso','shell','caltex','petrol','fuel'],
        'Entertainment':  ['netflix','spotify','youtube','cinema','steam','disney'],
        'Shopping':       ['daraz','telemart','shophive','amazon'],
        'Financial':      ['nayapay','jazzcash','easypaisa','hbl','meezan','alfalah','bank','ibft','raast'],
        'Gaming/Online':  ['skin.club','steam','epic','playstation','xbox'],
        'Income':         ['salary','ibft in','raast in','transfer from','riasat'],
    }
    for cat, words in kw.items():
        if any(w in m for w in words):
            return cat
    return 'Other'

PAKISTAN_MERCHANTS = {
    'Food & Dining':     ['Foodpanda', 'Careem Food', 'Cheetay', 'McDonald\'s PK', 'KFC Pakistan',
                          'Pizza Hut PK', 'Hardee\'s', 'Café Aylanto', 'Monal Restaurant', 'Burns Road Biryani'],
    'Groceries':         ['Imtiaz Supermarket', 'Carrefour PK', 'Chase Up', 'Metro Cash & Carry',
                          'Al-Fatah', 'Hyperstar', 'Naheed Supermarket'],
    'Transportation':    ['Careem', 'inDrive', 'Airlift', 'Bykea', 'PSO', 'Shell PK', 'Caltex',
                          'PTCL Broadband', 'Orange Line Metro'],
    'Utilities':         ['LESCO', 'GEPCO', 'SSGC', 'SNGPL', 'PTCL', 'Jazz', 'Telenor', 'Zong', 'Ufone'],
    'Subscriptions':     ['Netflix', 'Spotify', 'YouTube Premium', 'Apple iCloud', 'Adobe CC',
                          'Office 365', 'LinkedIn Premium', 'Canva Pro', 'Zoom', 'Dropbox'],
    'Shopping':          ['Daraz', 'Telemart', 'iShopping PK', 'Shophive', 'HomeShop', 'Yayvo'],
    'Healthcare':        ['Sehat Kahani', 'LabsDirect', 'Oladoc', 'Shifa Hospital', 'Agha Khan Lab'],
    'Entertainment':     ['Cinematix', 'Nueplex Cinemas', 'Cinepax', 'HBL PSL', 'Hamariweb Gaming'],
    'Education':         ['Coursera', 'Udemy', 'Edx', 'Quaid-i-Azam University', 'Virtual University'],
    'Financial':         ['NayaPay', 'JazzCash', 'Easypaisa', 'HBL Mobile', 'Meezan Bank', 'Bank Alfalah'],
}

def generate_realistic_dataset(months: int = 3, monthly_income: float = 80000) -> pd.DataFrame:
    """Generate a realistic Pakistani transaction dataset for demo/testing."""
    rng = np.random.default_rng(42)
    records = []
    base_date = pd.Timestamp.now() - pd.DateOffset(months=months)

    for month_offset in range(months):
        month_start = base_date + pd.DateOffset(months=month_offset)

        # Salary credit
        salary_day = rng.integers(1, 5)
        records.append({
            'date':             month_start + pd.DateOffset(days=int(salary_day)),
            'merchant_name':    'Employer Salary',
            'category':         'Income',
            'amount':           monthly_income * rng.uniform(0.97, 1.03),
            'transaction_type': 'income',
            'description':      'Monthly salary credit',
        })

        # Transactions per category
        category_budgets = {
            'Food & Dining': (monthly_income * 0.20, 8, 15),
            'Groceries':     (monthly_income * 0.10, 2, 6),
            'Transportation':(monthly_income * 0.08, 15, 30),
            'Utilities':     (monthly_income * 0.07, 1, 4),
            'Subscriptions': (monthly_income * 0.04, 3, 6),
            'Shopping':      (monthly_income * 0.12, 2, 8),
            'Healthcare':    (monthly_income * 0.03, 0, 2),
            'Entertainment': (monthly_income * 0.04, 1, 4),
            'Education':     (monthly_income * 0.02, 0, 2),
        }

        for cat, (budget, min_tx, max_tx) in category_budgets.items():
            n_tx = rng.integers(min_tx, max_tx + 1)
            if n_tx == 0:
                continue
            merchants_in_cat = PAKISTAN_MERCHANTS.get(cat, ['Unknown'])
            for _ in range(n_tx):
                day  = rng.integers(1, 29)
                merch = rng.choice(merchants_in_cat)
                amt  = max(50, rng.normal(budget / max(n_tx, 1), budget * 0.15 / max(n_tx, 1)))
                records.append({
                    'date':             month_start + pd.DateOffset(days=int(day)),
                    'merchant_name':    merch,
                    'category':         cat,
                    'amount':           round(float(amt), 2),
                    'transaction_type': 'expense',
                    'description':      f'{cat} payment – {merch}',
                })

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['transaction_id'] = [f'TXN{i:06d}' for i in range(len(df))]
    return df


# ─────────────────────────────────────────────
#  ENHANCED RECURRING PAYMENT DETECTION
# ─────────────────────────────────────────────

@_cache_data
def detect_recurring_payments_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    Improved subscription detector with fuzzy merchant matching
    and semantic grouping.
    """
    if len(df) == 0:
        return pd.DataFrame()

    expenses = df[df['transaction_type'] == 'expense'].copy()
    recurring = []

    for merchant in expenses['merchant_name'].unique():
        txns = expenses[expenses['merchant_name'] == merchant].sort_values('date')
        if len(txns) < 2:
            continue

        amounts  = txns['amount'].values
        mean_amt = amounts.mean()
        if mean_amt == 0:
            continue

        cv_amount = amounts.std() / mean_amt if mean_amt > 0 else 1.0

        interval_cv = 1.0
        if len(txns) >= 3:
            diffs = np.diff(txns['date'].values).astype('timedelta64[D]').astype(int)
            if len(diffs) > 0 and diffs.mean() > 0:
                interval_cv = diffs.std() / diffs.mean()

        if cv_amount <= 0.2 and interval_cv <= 0.4:
            # Estimate billing cycle
            if len(txns) >= 3:
                avg_days = np.diff(txns['date'].values).astype('timedelta64[D]').astype(float).mean()
                if avg_days < 10:
                    cycle = 'Weekly'
                elif avg_days < 20:
                    cycle = 'Bi-weekly'
                elif avg_days < 45:
                    cycle = 'Monthly'
                elif avg_days < 100:
                    cycle = 'Quarterly'
                else:
                    cycle = 'Annual'
            else:
                cycle = 'Monthly'

            annual = mean_amt * {'Weekly': 52, 'Bi-weekly': 26, 'Monthly': 12,
                                  'Quarterly': 4, 'Annual': 1}.get(cycle, 12)

            recurring.append({
                'merchant_name': merchant,
                'category':      txns['category'].iloc[0],
                'count':         len(txns),
                'avg_amount':    round(mean_amt, 2),
                'total_annual':  round(annual, 2),
                'billing_cycle': cycle,
                'last_charge':   txns['date'].max().strftime('%Y-%m-%d'),
                'consistency':   round((1 - cv_amount) * 100, 1),
            })

    if not recurring:
        return pd.DataFrame()

    return (pd.DataFrame(recurring)
              .sort_values('total_annual', ascending=False)
              .reset_index(drop=True))


# ─────────────────────────────────────────────
#  FINANCIAL HEALTH SCORE
# ─────────────────────────────────────────────

@_cache_data
def compute_financial_health_score(
    df: pd.DataFrame,
    income: float,
    expenses: float,
    recurring_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Composite financial health score (0–100) with sub-scores and recommendations.
    """
    scores = {}

    # 1. Savings ratio (max 30 pts)
    savings_rate = max(0, (income - expenses) / income * 100) if income > 0 else 0
    scores['savings'] = min(30, savings_rate * 1.5)

    # 2. Spending consistency (max 20 pts)
    exp_df = df[df['transaction_type'] == 'expense']
    if len(exp_df) >= 10:
        monthly = exp_df.groupby(exp_df['date'].dt.to_period('M'))['amount'].sum()
        cv = monthly.std() / monthly.mean() if monthly.mean() > 0 else 1.0
        scores['consistency'] = max(0, 20 - cv * 15)
    else:
        scores['consistency'] = 10

    # 3. Subscription burden (max 20 pts)
    if len(recurring_df) > 0 and income > 0:
        sub_ratio = recurring_df['avg_amount'].sum() / (income / 12) * 100
        scores['subscription'] = max(0, 20 - sub_ratio * 0.4)
    else:
        scores['subscription'] = 15

    # 4. Category diversity / risky spending (max 15 pts)
    if len(exp_df) > 0:
        cat_dist = exp_df.groupby('category')['amount'].sum() / exp_df['amount'].sum()
        risky_pct = cat_dist.get('Entertainment', 0) + cat_dist.get('Shopping', 0)
        scores['risk'] = max(0, 15 - risky_pct * 30)
    else:
        scores['risk'] = 10

    # 5. Income regularity (max 15 pts)
    inc_df = df[df['transaction_type'] == 'income']
    if len(inc_df) >= 2:
        inc_monthly = inc_df.groupby(inc_df['date'].dt.to_period('M'))['amount'].sum()
        inc_cv = inc_monthly.std() / inc_monthly.mean() if inc_monthly.mean() > 0 else 1.0
        scores['income_stability'] = max(0, 15 - inc_cv * 10)
    else:
        scores['income_stability'] = 7

    total = sum(scores.values())
    grade = ('A' if total >= 85 else 'B' if total >= 70 else
             'C' if total >= 55 else 'D' if total >= 40 else 'F')

    recs = []
    if scores['savings'] < 15:
        recs.append("Increase savings to at least 20% of income.")
    if scores['subscription'] < 10:
        recs.append("Subscription burden is high — review and cancel unused services.")
    if scores['risk'] < 7:
        recs.append("Reduce discretionary spending on entertainment and shopping.")
    if scores['consistency'] < 10:
        recs.append("Spending is highly variable — consider a monthly budget.")
    if not recs:
        recs.append("Excellent financial habits! Keep growing your emergency fund.")

    return {
        'total_score':   round(total, 1),
        'grade':         grade,
        'sub_scores':    {k: round(v, 1) for k, v in scores.items()},
        'savings_rate':  round(savings_rate, 1),
        'recommendations': recs,
        '_health_score': f"{round(total)}/100",
        '_assessment':   f"Grade {grade} — Savings rate {savings_rate:.1f}%",
        '_recommendations': " | ".join(recs[:2]),
    }


# ─────────────────────────────────────────────
#  SMART ALERTS ENGINE
# ─────────────────────────────────────────────

@_cache_data
def generate_smart_alerts(
    df: pd.DataFrame,
    recurring_df: pd.DataFrame,
    fraud_df: pd.DataFrame,
) -> List[Dict]:
    """Return a list of actionable financial alerts."""
    alerts = []

    # Alert 1: Fraud flagged
    fraud_count = int(fraud_df['is_fraud'].sum()) if 'is_fraud' in fraud_df.columns else 0
    if fraud_count > 0:
        alerts.append({
            'type': 'danger',
            'icon': 'alert-octagon',
            'title': 'Suspicious Transactions Detected',
            'message': f'{fraud_count} transaction(s) flagged as potentially fraudulent.',
            'action': 'Review Fraud Detection page',
        })

    # Alert 2: Month-over-month spending spike
    exp_df = df[df['transaction_type'] == 'expense']
    if len(exp_df) > 0:
        monthly = exp_df.groupby(exp_df['date'].dt.to_period('M'))['amount'].sum()
        if len(monthly) >= 2:
            last  = monthly.iloc[-1]
            prev  = monthly.iloc[-2]
            change_pct = ((last - prev) / prev * 100) if prev > 0 else 0
            if change_pct > 20:
                alerts.append({
                    'type': 'warning',
                    'icon': 'arrow-up-right',
                    'title': 'Spending Spike Detected',
                    'message': f'Spending increased by {change_pct:.0f}% compared to last month.',
                    'action': 'Review Analytics page',
                })
            elif change_pct < -20:
                alerts.append({
                    'type': 'success',
                    'icon': 'arrow-down-right',
                    'title': 'Spending Down',
                    'message': f'Spending decreased by {abs(change_pct):.0f}% vs last month.',
                    'action': 'Keep it up',
                })

    # Alert 3: Underused subscriptions
    if len(recurring_df) > 0:
        max_count = recurring_df['count'].max()
        underused = recurring_df[recurring_df['count'] < max_count * 0.4]
        if len(underused) > 0:
            names = ", ".join(underused['merchant_name'].head(3).tolist())
            alerts.append({
                'type': 'warning',
                'icon': 'repeat',
                'title': 'Underused Subscriptions',
                'message': f'{len(underused)} subscription(s) appear underused: {names}',
                'action': 'Review Recurring Payments page',
            })

    # Alert 4: Savings rate too low
    total_inc = df[df['transaction_type'] == 'income']['amount'].sum()
    total_exp = exp_df['amount'].sum()
    if total_inc > 0:
        sr = (total_inc - total_exp) / total_inc * 100
        if sr < 5:
            alerts.append({
                'type': 'danger',
                'icon': 'alert-triangle',
                'title': 'Very Low Savings Rate',
                'message': f'Savings rate is only {sr:.1f}%. Aim for 20%+.',
                'action': 'Review Budget Planning page',
            })

    if not alerts:
        alerts.append({
            'type': 'success',
            'icon': 'check-circle',
            'title': 'All Clear',
            'message': 'No financial alerts at this time.',
            'action': '',
        })

    return alerts


# ─────────────────────────────────────────────
#  PRESERVED ORIGINAL ai_utils.py FUNCTIONS
# ─────────────────────────────────────────────

@_cache_data
def forecast_spending_arima_style(df: pd.DataFrame, periods: int = 3) -> Dict[str, Any]:
    expenses = df[df['transaction_type'] == 'expense'].copy()
    monthly  = expenses.groupby(expenses['date'].dt.to_period('M'))['amount'].sum()
    if len(monthly) < 3:
        return {'error': 'Insufficient data'}
    y = monthly.values
    X = np.arange(len(y)).reshape(-1, 1)
    trend_model = LinearRegression()
    trend_model.fit(X, y)
    trend = trend_model.predict(X)
    slope = trend_model.coef_[0]
    detrended = y - trend
    alpha, smoothed = 0.3, np.zeros(len(detrended))
    smoothed[0] = detrended[0]
    for t in range(1, len(detrended)):
        smoothed[t] = alpha * detrended[t] + (1 - alpha) * smoothed[t - 1]
    forecasts = {}
    for i in range(1, periods + 1):
        future_trend = trend[-1] + slope * i
        seasonal     = smoothed[-1] * (0.9 ** i)
        month_idx    = len(monthly) + i - 1
        year         = monthly.index[0].year + (month_idx // 12)
        month        = (month_idx % 12) + 1
        forecasts[f'{year}-{month:02d}'] = max(0, future_trend + seasonal)
    std_error = np.std(y - trend)
    return {
        'forecasts':           {k: float(v) for k, v in forecasts.items()},
        'confidence_interval': f'±${std_error:,.0f}',
        'trend_direction':     'INCREASING' if slope > 0 else 'DECREASING',
        '_interpretation':     f"Spending trend: {'INCREASING' if slope > 0 else 'DECREASING'}",
    }

def predict_monthly_spending(df: pd.DataFrame, months_ahead: int = 3) -> Dict[str, Any]:
    return forecast_spending_arima_style(df, months_ahead)

def generate_spending_insights(df: pd.DataFrame, category: str = None) -> str:
    data = (df[df['category'] == category] if category
            else df[df['transaction_type'] == 'expense']).copy()
    if len(data) == 0:
        return "No data available"
    total = data['amount'].sum()
    avg   = data['amount'].mean()
    std   = data['amount'].std()
    top_m = data.groupby('merchant_name')['amount'].sum().nlargest(2)
    insights = []
    cv = (std / avg) if avg > 0 else 0
    insights.append(f"Spending consistency: CV={cv:.2f} ({'High variability' if cv > 1 else 'Stable'})")
    if len(top_m) > 0:
        pct = (top_m.iloc[0] / total) * 100
        insights.append(f"Top merchant {top_m.index[0]} is {pct:.1f}% of spending")
    outliers = data[data['amount'] > (avg + 3 * std)]
    if len(outliers) > 0:
        insights.append(f"Found {len(outliers)} outlier transactions")
    return " | ".join(insights)

def detect_anomalies_ai(df: pd.DataFrame) -> List[Dict[str, Any]]:
    expenses = df[df['transaction_type'] == 'expense'].copy()
    if len(expenses) < 10:
        return []
    avg = expenses['amount'].mean()
    std = expenses['amount'].std()
    threshold = avg + 2.5 * std
    anomalies = expenses[expenses['amount'] > threshold].copy()
    return [{'merchant': r['merchant_name'], 'amount': r['amount'],
             'z_score': (r['amount'] - avg) / std if std > 0 else 0}
            for _, r in anomalies.iterrows()]

def generate_budget_recommendations(df: pd.DataFrame, income: float) -> Dict[str, Any]:
    expenses = df[df['transaction_type'] == 'expense']
    cat_spend = expenses.groupby('category')['amount'].sum()
    total_exp = cat_spend.sum()
    savings = []
    for cat, amt in cat_spend.items():
        pct = (amt / income * 100) if income > 0 else 0
        benchmarks = {'Food & Dining': 15, 'Entertainment': 5, 'Shopping': 10, 'Transportation': 8}
        bm = benchmarks.get(cat, 10)
        if pct > bm:
            savings.append({'category': cat, 'potential_saving': round((pct - bm) / 100 * income, 2)})
    total_savings = sum(s['potential_saving'] for s in savings)
    return {'savings_opportunities': savings,
            'total_savings_potential': total_savings,
            '_ai_tips': f"Potential savings: ${total_savings:,.0f}"}

def analyze_subscriptions_ai(recurring_df: pd.DataFrame) -> Dict[str, Any]:
    result = {'total': len(recurring_df),
              'monthly_cost': float(recurring_df['avg_amount'].sum()) if len(recurring_df) > 0 else 0}
    if len(recurring_df) > 0:
        top3 = recurring_df.nlargest(3, 'total_annual')
        result['_savings_opportunities'] = f"Top: {', '.join(top3['merchant_name'].tolist())}"
        result['_alternatives'] = "Compare alternatives for 20-30% savings"
    return result

def generate_financial_health_assessment(df: pd.DataFrame, income: float, expenses: float) -> Dict[str, Any]:
    savings_rate = ((income - expenses) / income * 100) if income > 0 else 0
    health_score = min(100, max(0, (savings_rate / 20) * 100))
    return {
        '_health_score':    f"{int(health_score)}/100",
        '_assessment':      f"Savings rate: {savings_rate:.1f}%",
        '_recommendations': f"Focus on {'reducing discretionary' if expenses > income * 0.8 else 'optimization'} spending",
    }

def generate_smart_bill_reminders(recurring_df: pd.DataFrame, df: pd.DataFrame) -> Dict[str, Any]:
    if len(recurring_df) == 0:
        return {'error': 'No recurring payments'}
    total = recurring_df['avg_amount'].sum()
    return {
        'total_monthly':    float(total),
        '_recommendations': f"Total bills: ${total:,.0f}. Negotiate largest items.",
        '_urgency':         "Optimize payment schedule for cash flow management",
    }
