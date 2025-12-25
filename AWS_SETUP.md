# AWS Setup Instructions

## 1. Install Dependencies
```bash
pip install yfinance pandas pandas_ta ephem xgboost joblib
```

## 2. Configure Email (edit the script)
Open `daily_signal_scanner_aws.py` and scroll to the `CONFIG` section:
```python
CONFIG = {
    # AWS SES settings
    'smtp_server': 'email-smtp.us-east-1.amazonaws.com',
    'smtp_port': 587,
    'smtp_username': 'YOUR_SES_SMTP_USERNAME',
    'smtp_password': 'YOUR_SES_SMTP_PASSWORD',
    'email_from': 'bulletproof@yourdomain.com',
    'email_to': ['your-email@gmail.com'],
    ...
}
```

Or use environment variables:
```bash
export SMTP_USERNAME='your_ses_username'
export SMTP_PASSWORD='your_ses_password'
```

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
