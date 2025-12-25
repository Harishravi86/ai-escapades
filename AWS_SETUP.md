# AWS Setup Instructions

## 1. Install Dependencies
```bash
pip install yfinance pandas pandas_ta ephem xgboost joblib
```

## 2. Configure Email
### Option A: Gmail (Recommended/Easiest)
1.  Go to your Google Account > Security.
2.  Enable **2-Step Verification** if not already on.
3.  Go to **App passwords** (search for it in the top bar).
4.  Create a new app password:
## 2. Configure Email
### Option A: Gmail (Recommended/Easiest)
1.  Go to your Google Account > Security.
2.  Enable **2-Step Verification** if not already on.
3.  Go to **App passwords** (search for it in the top bar).
4.  Create a new app password:
    *   **App**: "Mail"
    *   **Device**: "Other (AWS Scanner)"
5.  Copy the 16-character code (e.g., `xxxx xxxx xxxx xxxx`).

### IMPORTANT: Security
Do **NOT** paste this password into the script if you plan to share it. Instead, set it as an environment variable:

**Linux / AWS (Bash):**
```bash
export GMAIL_APP_PASSWORD='your 16 char password'
```

**Windows (PowerShell):**
```powershell
$env:GMAIL_APP_PASSWORD='your 16 char password'
```

The script is configured to look for `GMAIL_APP_PASSWORD` automatically.

## 3. Setup Cron (3 PM CST = 21:00 UTC)
Open crontab:
```bash
crontab -e
```
Add this line (adjust path if needed):
```
0 21 * * 1-5 /usr/bin/python3 /home/ubuntu/daily_signal_scanner_aws.py >> /var/log/bulletproof.log 2>&1
```

## 4. Test
Test email delivery:
```bash
python3 daily_signal_scanner_aws.py --test-email
```

Run full scan:
```bash
python3 daily_signal_scanner_aws.py
```
