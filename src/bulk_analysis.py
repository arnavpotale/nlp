"""
SMS Shield 2.0 - Bulk SMS Classification & Reporting Engine
Author: SMS Shield NLP Project

This module provides batch processing, validation, filtering, and export
capabilities for bulk SMS CSV files.
"""

from io import StringIO, BytesIO
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from src.prediction import SMSShieldPredictor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = PROJECT_ROOT / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_and_validate_csv(file_buffer) -> Tuple[pd.DataFrame, List[str]]:
    """
    Reads an uploaded CSV file buffer and returns the DataFrame and list of candidate text columns.
    Defensively handles encoding and formatting errors.

    Parameters
    ----------
    file_buffer : file-like object or bytes
        Uploaded CSV data.

    Returns
    -------
    (pd.DataFrame, list of str)
        DataFrame and list of candidate column names.
    """
    try:
        df = pd.read_csv(file_buffer)
    except UnicodeDecodeError:
        if hasattr(file_buffer, "seek"):
            file_buffer.seek(0)
        df = pd.read_csv(file_buffer, encoding="latin-1")

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    # Identify potential text/message columns
    text_cols = [c for c in df.columns if df[c].dtype == "object" or "message" in str(c).lower() or "sms" in str(c).lower() or "text" in str(c).lower()]
    if not text_cols:
        text_cols = list(df.columns)

    return df, text_cols


def run_bulk_classification(
    df: pd.DataFrame,
    message_col: str,
    predictor: Optional[SMSShieldPredictor] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes 3-model classification across all rows in the dataframe.
    Never silently discards invalid rows; flags empty/malformed rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    message_col : str
        Name of the column containing SMS messages.
    predictor : SMSShieldPredictor, optional
        Pre-instantiated predictor instance.

    Returns
    -------
    (pd.DataFrame, dict)
        Enriched results dataframe and summary statistics dictionary.
    """
    if message_col not in df.columns:
        raise KeyError(f"Selected column '{message_col}' not found in the uploaded file.")

    if predictor is None:
        predictor = SMSShieldPredictor().load()

    raw_messages = df[message_col].fillna("").astype(str).tolist()
    results_df = predictor.predict_batch(raw_messages)

    # Combine with original metadata columns (if any exist beyond message_col)
    original_other_cols = [c for c in df.columns if c != message_col]
    if original_other_cols:
        for c in original_other_cols:
            results_df[c] = df[c].values

    # Compute summary statistics
    total_messages = len(results_df)
    empty_count = int(results_df["Is_Empty_Row"].sum())
    valid_count = total_messages - empty_count

    spam_count = int((results_df["Consensus_Prediction"] == "SPAM").sum())
    ham_count = int((results_df["Consensus_Prediction"] == "HAM").sum())
    spam_pct = (spam_count / total_messages * 100) if total_messages > 0 else 0.0

    # Model agreement stats
    unanimous_count = int(results_df["Spam_Votes"].isin(["3/3", "0/3"]).sum())
    divided_count = total_messages - unanimous_count
    unanimous_pct = (unanimous_count / total_messages * 100) if total_messages > 0 else 0.0

    # Per-model counts
    nb_spam = int((results_df["NB_Prediction"] == "SPAM").sum())
    lr_spam = int((results_df["LR_Prediction"] == "SPAM").sum())
    svm_spam = int((results_df["SVM_Prediction"] == "SPAM").sum())

    summary = {
        "total_messages": total_messages,
        "valid_count": valid_count,
        "empty_count": empty_count,
        "spam_count": spam_count,
        "ham_count": ham_count,
        "spam_percentage": spam_pct,
        "unanimous_consensus_count": unanimous_count,
        "unanimous_percentage": unanimous_pct,
        "divided_consensus_count": divided_count,
        "nb_spam_count": nb_spam,
        "lr_spam_count": lr_spam,
        "svm_spam_count": svm_spam,
    }

    return results_df, summary


def generate_sample_bulk_csv(save_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Generates a realistic test dataset of 25 sample SMS messages
    spanning phishing scams, promotions, OTP alerts, and conversational ham.
    """
    sample_data = [
        {"sms_id": 1, "text": "WINNER! You've been selected for a £1,000 cash prize! Text CLAIM to 80808 now. T&Cs apply."},
        {"sms_id": 2, "text": "Hey, are you free for coffee tomorrow afternoon around 3?"},
        {"sms_id": 3, "text": "URGENT! Your bank account has unusual activity. Call 08001234567 immediately to verify."},
        {"sms_id": 4, "text": "Don't forget we have project presentation at 10 AM on Monday."},
        {"sms_id": 5, "text": "Congratulations! You won free tickets to the championship game! Dial 09061701461 to claim."},
        {"sms_id": 6, "text": "Can you please send me the class notes from yesterday's NLP lecture?"},
        {"sms_id": 7, "text": "FreeMsg: 500 free ringtones waiting for you. Text YES to 88088 now! £1.50 per week."},
        {"sms_id": 8, "text": "I'm running about 10 minutes late, please grab a table inside."},
        {"sms_id": 9, "text": "ALERT: Your delivery parcel #94821 is on hold. Visit http://parcel-track-update.com/claim"},
        {"sms_id": 10, "text": "Ok perfect, I will see you at the library soon."},
        {"sms_id": 11, "text": "PRIVATE! Your account is credited with 500 bonus points. Call 08712460324 to redeem."},
        {"sms_id": 12, "text": "Did you finish the assignment for machine learning class?"},
        {"sms_id": 13, "text": "GUARANTEED £500 loan approved in 5 minutes! No credit check required. Text CASH to 78877."},
        {"sms_id": 14, "text": "Thanks for dinner yesterday, it was wonderful catching up!"},
        {"sms_id": 15, "text": "You have an unread message from secret admirer. Text MATCH to 69888 to view. £2/txt"},
        {"sms_id": 16, "text": "Your OTP for transaction at Amazon is 482910. Valid for 5 minutes. Do not share."},
        {"sms_id": 17, "text": "EXCLUSIVE OFFER! Upgrade to the newest smartphone for FREE! Call 08002986030 now."},
        {"sms_id": 18, "text": "Please bring the car keys when you come downstairs."},
        {"sms_id": 19, "text": "FINAL NOTICE: Claim your £250 voucher before midnight tonight. Code: VK981."},
        {"sms_id": 20, "text": "Mom called earlier, she asked if you can visit this weekend."},
        {"sms_id": 21, "text": ""},  # Intentional empty row for validation test
        {"sms_id": 22, "text": "HOT LIVE chat with local singles in your area! Call 09099726392 now. 18+ only."},
        {"sms_id": 23, "text": "Happy birthday! Hope you have a fantastic day ahead!"},
        {"sms_id": 24, "text": "URGENT NOTICE: IRS tax refund of $480 is pending your confirmation. Click here to confirm."},
        {"sms_id": 25, "text": "Will be home by 7 PM tonight, love you!"},
    ]

    df = pd.DataFrame(sample_data)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

    return df

