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
    *   **App**: "Mail"
    *   **Device**: "Other (AWS Scanner)"
5.  Copy the 16-character code (e.g., `xxxx xxxx xxxx xxxx`).
6.  Edit `daily_signal_scanner_aws.py` and paste it into the `CONFIG` section.

### Option B: AWS SES (Production)
If using AWS Simple Email Service:
1.  Verify your sender email and domain in the SES Console.
2.  Create SMTP credentials in IAM.
3.  Use the `email-smtp` server settings in the script.

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
