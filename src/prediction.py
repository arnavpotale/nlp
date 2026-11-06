"""
SMS Shield 2.0 - Unified Model Prediction & Consensus Engine
Author: SMS Shield NLP Project

This module provides unified inference across all three machine learning classifiers:
1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine (LinearSVC with Platt-calibrated probabilities)

Features:
- Single-message prediction with full probability breakdowns and consensus voting.
- Batch prediction optimized for Bulk CSV analysis.
- Defensive handling of empty strings, whitespace, and out-of-vocabulary inputs.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import preprocess_text, preprocess_with_steps
from src.explainability import extract_heuristic_signals, explain_message_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class SMSShieldPredictor:
    """
    Unified predictor encapsulating TF-IDF vectorization and the three ML models.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.vectorizer = None
        self.nb_model = None
        self.lr_model = None
        self.svc_raw = None
        self.svc_calibrated = None
        self._loaded = False

    def load(self):
        """Loads all persisted model artifacts from disk."""
        vec_path = self.models_dir / "tfidf_vectorizer.joblib"
        nb_path = self.models_dir / "multinomial_nb.joblib"
        lr_path = self.models_dir / "logistic_regression.joblib"
        svc_path = self.models_dir / "linear_svc.joblib"
        svc_cal_path = self.models_dir / "calibrated_linear_svc.joblib"

        required = [vec_path, nb_path, lr_path]
        missing = [str(p.name) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required model artifacts: {', '.join(missing)}. "
                "Please run `python src/train.py` first."
            )

        self.vectorizer = joblib.load(vec_path)
        self.nb_model = joblib.load(nb_path)
        self.lr_model = joblib.load(lr_path)

        if svc_path.exists():
            self.svc_raw = joblib.load(svc_path)
        if svc_cal_path.exists():
            self.svc_calibrated = joblib.load(svc_cal_path)
        elif self.svc_raw is not None:
            # Fallback to raw svc if calibrated version isn't present
            self.svc_calibrated = self.svc_raw

        self._loaded = True
        return self

    def is_loaded(self) -> bool:
        return self._loaded

    def predict_single(self, raw_text: str) -> Dict[str, Any]:
        """
        Executes preprocessing, heuristic analysis, and inference across all three models.

        Parameters
        ----------
        raw_text : str
            The input SMS text string.

        Returns
        -------
        dict
            Comprehensive prediction dictionary.
        """
        if not self._loaded:
            self.load()

        # Handle empty/whitespace input
        if not isinstance(raw_text, str) or not raw_text.strip():
            return {
                "is_valid": False,
                "error": "Message is empty or contains only whitespace.",
                "raw_text": raw_text or "",
                "clean_text": "",
                "heuristics": extract_heuristic_signals(""),
            }

        # 1. NLP Preprocessing
        step_details = preprocess_with_steps(raw_text)
        clean_text = step_details["final_preprocessed"]

        # 2. Heuristic Signals
        heuristics = extract_heuristic_signals(raw_text)

        # 3. TF-IDF Vectorization
        input_vec = self.vectorizer.transform([clean_text])

        # 4. Model Predictions & Probabilities
        # Naive Bayes
        nb_pred = int(self.nb_model.predict(input_vec)[0])
        nb_proba = self.nb_model.predict_proba(input_vec)[0]  # [P(ham), P(spam)]

        # Logistic Regression
        lr_pred = int(self.lr_model.predict(input_vec)[0])
        lr_proba = self.lr_model.predict_proba(input_vec)[0]  # [P(ham), P(spam)]

        # Linear SVM
        if self.svc_calibrated is not None and hasattr(self.svc_calibrated, "predict_proba"):
            svc_pred = int(self.svc_calibrated.predict(input_vec)[0])
            svc_proba = self.svc_calibrated.predict_proba(input_vec)[0]
        elif self.svc_raw is not None:
            svc_pred = int(self.svc_raw.predict(input_vec)[0])
            # Sigmoid approximation of decision function if uncalibrated
            d = float(self.svc_raw.decision_function(input_vec)[0])
            p_spam = 1.0 / (1.0 + np.exp(-d))
            svc_proba = np.array([1.0 - p_spam, p_spam])
        else:
            svc_pred = lr_pred
            svc_proba = lr_proba

        decision_dist = (
            float(self.svc_raw.decision_function(input_vec)[0])
            if (self.svc_raw is not None and hasattr(self.svc_raw, "decision_function"))
            else 0.0
        )

        # 5. Consensus Calculation
        votes = [nb_pred, lr_pred, svc_pred]
        spam_votes = sum(votes)
        ham_votes = 3 - spam_votes

        if spam_votes == 3:
            consensus = "SPAM"
            consensus_detail = "Unanimous Spam (3/3 models)"
            consensus_color = "red"
        elif spam_votes == 2:
            consensus = "SPAM"
            consensus_detail = "Majority Spam (2/3 models)"
            consensus_color = "orange"
        elif spam_votes == 1:
            consensus = "HAM"
            consensus_detail = "Majority Ham (2/3 models)"
            consensus_color = "blue"
        else:
            consensus = "HAM"
            consensus_detail = "Unanimous Ham (3/3 models)"
            consensus_color = "green"

        # 6. Feature Contribution Explanation
        feature_explanation_df = explain_message_features(
            clean_text=clean_text,
            vectorizer=self.vectorizer,
            lr_model=self.lr_model,
            nb_model=self.nb_model,
            svc_model=self.svc_raw or self.lr_model,
        )

        return {
            "is_valid": True,
            "raw_text": raw_text,
            "clean_text": clean_text,
            "step_details": step_details,
            "heuristics": heuristics,
            "models": {
                "naive_bayes": {
                    "name": "Multinomial Naive Bayes",
                    "pred": nb_pred,
                    "label": "SPAM" if nb_pred == 1 else "HAM",
                    "prob_spam": float(nb_proba[1]),
                    "prob_ham": float(nb_proba[0]),
                },
                "logistic_regression": {
                    "name": "Logistic Regression",
                    "pred": lr_pred,
                    "label": "SPAM" if lr_pred == 1 else "HAM",
                    "prob_spam": float(lr_proba[1]),
                    "prob_ham": float(lr_proba[0]),
                },
                "linear_svm": {
                    "name": "Linear SVM",
                    "pred": svc_pred,
                    "label": "SPAM" if svc_pred == 1 else "HAM",
                    "prob_spam": float(svc_proba[1]),
                    "prob_ham": float(svc_proba[0]),
                    "decision_function": decision_dist,
                },
            },
            "consensus": consensus,
            "consensus_detail": consensus_detail,
            "consensus_color": consensus_color,
            "spam_votes": spam_votes,
            "ham_votes": ham_votes,
            "feature_explanation": feature_explanation_df,
        }

    def predict_batch(self, messages: List[str]) -> pd.DataFrame:
        """
        Classifies a batch of messages for Bulk SMS analysis.
        Handles invalid / empty rows without silently dropping them.

        Parameters
        ----------
        messages : list of str
            List of raw SMS message strings.

        Returns
        -------
        pd.DataFrame
            DataFrame with predictions and probabilities from all 3 models.
        """
        if not self._loaded:
            self.load()

        records = []
        cleaned_list = []
        valid_indices = []

        # Preprocess each row defensively
        for idx, text in enumerate(messages):
            if not isinstance(text, str) or not text.strip():
                cleaned_list.append("")
            else:
                c = preprocess_text(text)
                cleaned_list.append(c)
                valid_indices.append(idx)

        # Batch vectorization
        vec_matrix = self.vectorizer.transform(cleaned_list)

        nb_preds = self.nb_model.predict(vec_matrix)
        nb_probas = self.nb_model.predict_proba(vec_matrix)[:, 1]

        lr_preds = self.lr_model.predict(vec_matrix)
        lr_probas = self.lr_model.predict_proba(vec_matrix)[:, 1]

        if self.svc_calibrated is not None and hasattr(self.svc_calibrated, "predict_proba"):
            svc_preds = self.svc_calibrated.predict(vec_matrix)
            svc_probas = self.svc_calibrated.predict_proba(vec_matrix)[:, 1]
        elif self.svc_raw is not None:
            svc_preds = self.svc_raw.predict(vec_matrix)
            d = self.svc_raw.decision_function(vec_matrix)
            svc_probas = 1.0 / (1.0 + np.exp(-d))
        else:
            svc_preds = lr_preds
            svc_probas = lr_probas

        for i, text in enumerate(messages):
            raw = str(text) if text is not None else ""
            is_empty = not bool(raw.strip())
            nb_p = int(nb_preds[i])
            lr_p = int(lr_preds[i])
            svc_p = int(svc_preds[i])

            spam_votes = nb_p + lr_p + svc_p
            consensus = "SPAM" if spam_votes >= 2 else "HAM"

            records.append({
                "Message_ID": i + 1,
                "Message": raw,
                "Cleaned_Text": cleaned_list[i],
                "Character_Count": len(raw),
                "Word_Count": len(raw.split()) if raw else 0,
                "NB_Prediction": "SPAM" if nb_p == 1 else "HAM",
                "NB_Spam_Probability": round(float(nb_probas[i]), 4),
                "LR_Prediction": "SPAM" if lr_p == 1 else "HAM",
                "LR_Spam_Probability": round(float(lr_probas[i]), 4),
                "SVM_Prediction": "SPAM" if svc_p == 1 else "HAM",
                "SVM_Spam_Probability": round(float(svc_probas[i]), 4),
                "Consensus_Prediction": consensus,
                "Spam_Votes": f"{spam_votes}/3",
                "Is_Empty_Row": is_empty,
            })

        return pd.DataFrame(records)
