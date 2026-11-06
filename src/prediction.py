"""
SMS Shield 3.0 - Unified Multi-Feature Model Prediction & Consensus Engine
Author: SMS Shield NLP Project

This module provides unified inference across:
1. Feature Spaces:
   - Word-level TF-IDF (1,2 n-grams)
   - Character-level TF-IDF (char_wb 2,5 n-grams)
   - Combined Word + Character TF-IDF (FeatureUnion)
2. Classifiers:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear SVM (with Platt-calibrated probability estimation)
3. Defense Modes:
   - Raw input pipeline
   - Pre-normalization defense pipeline (reversing leetspeak, spacing, and delimiters)
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import preprocess_text, preprocess_with_steps
from src.normalization import normalize_sms
from src.explainability import extract_heuristic_signals, explain_message_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class SMSShieldPredictor:
    """
    Unified predictor supporting Word, Character, and Combined TF-IDF models,
    with optional SMS text normalization defense.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR

        # Word models
        self.vectorizer = None
        self.nb_model = None
        self.lr_model = None
        self.svc_raw = None
        self.svc_calibrated = None

        # Character models
        self.char_vectorizer = None
        self.char_nb = None
        self.char_lr = None
        self.char_svc = None

        # Combined models
        self.comb_vectorizer = None
        self.comb_nb = None
        self.comb_lr = None
        self.comb_svc = None

        self._loaded = False

    def load(self):
        """Loads all persisted model artifacts from disk."""
        # 1. Load Word models
        vec_path = self.models_dir / "tfidf_vectorizer.joblib"
        nb_path = self.models_dir / "multinomial_nb.joblib"
        lr_path = self.models_dir / "logistic_regression.joblib"
        svc_path = self.models_dir / "linear_svc.joblib"
        svc_cal_path = self.models_dir / "calibrated_linear_svc.joblib"

        if not vec_path.exists():
            raise FileNotFoundError(f"Missing vectorizer: {vec_path}. Run python src/train.py.")

        self.vectorizer = joblib.load(vec_path)
        self.nb_model = joblib.load(nb_path)
        self.lr_model = joblib.load(lr_path)
        self.svc_raw = joblib.load(svc_path) if svc_path.exists() else None
        self.svc_calibrated = joblib.load(svc_cal_path) if svc_cal_path.exists() else self.svc_raw

        # 2. Load Character models (if available)
        char_vec_path = self.models_dir / "char_tfidf_vectorizer.joblib"
        if char_vec_path.exists():
            self.char_vectorizer = joblib.load(char_vec_path)
            self.char_nb = joblib.load(self.models_dir / "char_nb.joblib")
            self.char_lr = joblib.load(self.models_dir / "char_lr.joblib")
            self.char_svc = joblib.load(self.models_dir / "char_svc.joblib")

        # 3. Load Combined models (if available)
        comb_vec_path = self.models_dir / "combined_tfidf_vectorizer.joblib"
        if comb_vec_path.exists():
            self.comb_vectorizer = joblib.load(comb_vec_path)
            self.comb_nb = joblib.load(self.models_dir / "comb_nb.joblib")
            self.comb_lr = joblib.load(self.models_dir / "comb_lr.joblib")
            self.comb_svc = joblib.load(self.models_dir / "comb_svc.joblib")

        self._loaded = True
        return self

    def is_loaded(self) -> bool:
        return self._loaded

    def predict_single(
        self,
        raw_text: str,
        use_normalization: bool = False,
        feature_space: str = "word"
    ) -> Dict[str, Any]:
        """
        Executes prediction on a single SMS message.

        Parameters
        ----------
        raw_text : str
            The input SMS text.
        use_normalization : bool
            Whether to apply SMS text normalization before feature extraction.
        feature_space : str
            One of 'word', 'char', or 'combined'.

        Returns
        -------
        dict
            Prediction verdict, probabilities, heuristics, and normalization logs.
        """
        if not self._loaded:
            self.load()

        if not isinstance(raw_text, str) or not raw_text.strip():
            return {
                "is_valid": False,
                "error": "Message is empty or contains only whitespace.",
                "raw_text": raw_text or "",
                "clean_text": "",
                "heuristics": extract_heuristic_signals(""),
            }

        # 1. Optional Normalization Defense
        norm_result = normalize_sms(raw_text) if use_normalization else None
        active_raw = norm_result["normalized_text"] if (use_normalization and norm_result) else raw_text

        # 2. Preprocessing
        step_details = preprocess_with_steps(active_raw)
        clean_text = step_details["final_preprocessed"]

        # 3. Heuristic Signals
        heuristics = extract_heuristic_signals(raw_text)

        # 4. Select Vectorizer & Models based on feature_space
        if feature_space == "char" and self.char_vectorizer is not None:
            vec = self.char_vectorizer
            nb = self.char_nb
            lr = self.char_lr
            svc = self.char_svc
        elif feature_space == "combined" and self.comb_vectorizer is not None:
            vec = self.comb_vectorizer
            nb = self.comb_nb
            lr = self.comb_lr
            svc = self.comb_svc
        else:
            vec = self.vectorizer
            nb = self.nb_model
            lr = self.lr_model
            svc = self.svc_calibrated or self.svc_raw

        input_vec = vec.transform([clean_text])

        # 5. Model Inference
        # Naive Bayes
        nb_pred = int(nb.predict(input_vec)[0])
        nb_proba = nb.predict_proba(input_vec)[0] if hasattr(nb, "predict_proba") else np.array([0.5, 0.5])

        # Logistic Regression
        lr_pred = int(lr.predict(input_vec)[0])
        lr_proba = lr.predict_proba(input_vec)[0] if hasattr(lr, "predict_proba") else np.array([0.5, 0.5])

        # Linear SVM
        if hasattr(svc, "predict_proba"):
            svc_pred = int(svc.predict(input_vec)[0])
            svc_proba = svc.predict_proba(input_vec)[0]
        else:
            svc_pred = int(svc.predict(input_vec)[0])
            d = float(svc.decision_function(input_vec)[0])
            p_spam = 1.0 / (1.0 + np.exp(-d))
            svc_proba = np.array([1.0 - p_spam, p_spam])

        # Consensus
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

        # Feature explanation (for word models)
        feat_df = pd.DataFrame()
        if feature_space == "word":
            feat_df = explain_message_features(
                clean_text=clean_text,
                vectorizer=self.vectorizer,
                lr_model=self.lr_model,
                nb_model=self.nb_model,
                svc_model=self.svc_raw or self.lr_model,
            )

        return {
            "is_valid": True,
            "raw_text": raw_text,
            "active_text": active_raw,
            "clean_text": clean_text,
            "use_normalization": use_normalization,
            "feature_space": feature_space,
            "normalization": norm_result,
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
                },
            },
            "consensus": consensus,
            "consensus_detail": consensus_detail,
            "consensus_color": consensus_color,
            "spam_votes": spam_votes,
            "ham_votes": ham_votes,
            "feature_explanation": feat_df,
        }

    def predict_batch(
        self,
        messages: List[str],
        use_normalization: bool = False,
        feature_space: str = "word"
    ) -> pd.DataFrame:
        """
        Classifies a batch of messages for Bulk SMS analysis.
        """
        if not self._loaded:
            self.load()

        records = []
        cleaned_list = []

        for text in messages:
            if not isinstance(text, str) or not text.strip():
                cleaned_list.append("")
            else:
                target_text = normalize_sms(text)["normalized_text"] if use_normalization else text
                c = preprocess_text(target_text)
                cleaned_list.append(c)

        # Select vectorizer and models
        if feature_space == "char" and self.char_vectorizer is not None:
            vec = self.char_vectorizer
            nb = self.char_nb
            lr = self.char_lr
            svc = self.char_svc
        elif feature_space == "combined" and self.comb_vectorizer is not None:
            vec = self.comb_vectorizer
            nb = self.comb_nb
            lr = self.comb_lr
            svc = self.comb_svc
        else:
            vec = self.vectorizer
            nb = self.nb_model
            lr = self.lr_model
            svc = self.svc_calibrated or self.svc_raw

        vec_matrix = vec.transform(cleaned_list)

        nb_preds = nb.predict(vec_matrix)
        nb_probas = nb.predict_proba(vec_matrix)[:, 1] if hasattr(nb, "predict_proba") else np.zeros(len(messages))

        lr_preds = lr.predict(vec_matrix)
        lr_probas = lr.predict_proba(vec_matrix)[:, 1] if hasattr(lr, "predict_proba") else np.zeros(len(messages))

        svc_preds = svc.predict(vec_matrix)
        if hasattr(svc, "predict_proba"):
            svc_probas = svc.predict_proba(vec_matrix)[:, 1]
        else:
            d = svc.decision_function(vec_matrix)
            svc_probas = 1.0 / (1.0 + np.exp(-d))

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
