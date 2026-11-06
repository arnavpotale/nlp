"""
SMS Shield 3.0 - SMS Text Normalization & Obfuscation Reversal Module
Author: SMS Shield NLP Project

This module provides deterministic and rule-based text normalization to counteract
common SMS obfuscation, adversarial evasion, and informal text modifications:
1. Number substitutions (Leetspeak numbers: FR33 -> FREE, B0NU5 -> BONUS, W1NNER -> WINNER).
2. Symbol substitutions (W!NNER -> WINNER, C@SH -> CASH, PR!ZE -> PRIZE).
3. Repeated character reduction (freeeee -> free, cooooool -> cool).
4. Common SMS abbreviations (u -> you, ur -> your, gr8 -> great, txt -> text, b4 -> before).
5. Spaced-out and delimited word reassembly (w i n n e r -> winner, w*i*n*n*e*r -> winner).

CRITICAL DESIGN PRINCIPLE:
Does not blindly replace characters in ways that alter legitimate data (such as telephone
numbers, dates, times, monetary quantities, or valid numeric values).
"""

import re
from typing import Dict, List, Any, Tuple, Optional


# Mapping for leetspeak numbers inside words
LEET_NUMBER_MAP = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
}

# Mapping for evasive symbol substitutions inside words
SYMBOL_SUB_MAP = {
    "@": "a",
    "$": "s",
    "!": "i",
    "€": "e",
    "£": "e",  # only when embedded inside alphabetic tokens like 'FR££'
}

# Common SMS and chat abbreviations
SMS_ABBREVIATIONS = {
    "u": "you",
    "ur": "your",
    "urs": "yours",
    "r": "are",
    "c": "see",
    "b4": "before",
    "gr8": "great",
    "txt": "text",
    "msg": "message",
    "plz": "please",
    "pls": "please",
    "thx": "thanks",
    "tnx": "thanks",
    "wkly": "weekly",
    "comp": "competition",
    "tkts": "tickets",
    "tkt": "ticket",
    "rcv": "receive",
    "wat": "what",
    "dun": "do not",
    "wan2": "want to",
    "w/e": "weekend",
    "idk": "i do not know",
    "omw": "on my way",
    "atm": "at the moment",
    "btw": "by the way",
    "imho": "in my opinion",
    "fyi": "for your information",
}

# Known common spam keywords for reassembly verification
TARGET_SPAM_WORDS = {
    "free", "winner", "win", "won", "prize", "cash", "claim", "urgent",
    "bonus", "award", "reward", "call", "text", "loan", "credit", "account",
    "refund", "guaranteed", "selected", "entry", "voucher", "mobile", "service"
}


def _split_affixes(word: str) -> Tuple[str, str, str]:
    """
    Separates outer leading and trailing punctuation from the core word.
    Example: 'W!NNER!!' -> ('', 'W!NNER', '!!')
    """
    # Match leading non-alphanumeric characters (excluding currency symbols attached to digits)
    m_lead = re.match(r"^[^\w£$€]+", word)
    lead = m_lead.group(0) if m_lead else ""
    remainder = word[len(lead):]

    # Match trailing non-alphanumeric punctuation
    m_trail = re.search(r"[^\w]+$", remainder)
    trail = m_trail.group(0) if m_trail else ""
    core = remainder[:-len(trail)] if trail else remainder

    return lead, core, trail


def _is_protected_numeric(core: str) -> bool:
    """
    Checks if a token is a legitimate numeric entity that must NOT be altered:
    - Phone numbers (e.g. 09061701461, 87121)
    - Currency amounts (e.g. £900, $500, 100k)
    - Times, years, and dates (e.g. 5pm, 2024, 12hr)
    """
    # Standalone digits or phone numbers
    if core.isdigit() and len(core) >= 4:
        return True
    # Monetary amounts like £900, $500, 500k
    if re.match(r"^[£$€]\d+[\d,.]*(?:k|m|bn)?$", core, re.IGNORECASE):
        return True
    # Number with units: 12hr, 10min, 5pm, 24/7
    if re.match(r"^\d+(?:am|pm|hr|hrs|min|mins|sec|k|m|th|st|nd|rd)$", core, re.IGNORECASE):
        return True
    return False


def normalize_repeated_characters(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Condenses 3 or more consecutive identical letters down to 2 or 1.
    Example: 'freeeee' -> 'free', 'cooooool' -> 'cool', 'hurryyyy' -> 'hurry'.
    """
    mods = []

    def repl(match):
        full_match = match.group(0)
        char = match.group(1)
        # For 'e', 'o', 'l', 's' keeping 2 is natural (free, cool, bell, kiss)
        if char.lower() in {"e", "o", "l", "s", "f", "p"}:
            replacement = char * 2
        else:
            replacement = char
        if full_match != replacement:
            mods.append({
                "type": "Repeated Characters",
                "original": full_match,
                "normalized": replacement
            })
        return replacement

    # Matches 3 or more consecutive identical letters
    pattern = re.compile(r"([a-zA-Z])\1{2,}")
    normalized = pattern.sub(repl, text)
    return normalized, mods


def normalize_spaced_words(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Reassembles deliberately spaced-out words designed to bypass tokenizers.
    Example: 'w i n n e r' -> 'winner', 'f r e e' -> 'free'.
    """
    mods = []

    def repl(match):
        full_match = match.group(0)
        letters = full_match.split()
        rejoined = "".join(letters)
        # Verify length and check if it forms an alphabetic sequence >= 3 chars
        if len(rejoined) >= 3 and rejoined.isalpha():
            mods.append({
                "type": "Spaced Word Reassembly",
                "original": full_match,
                "normalized": rejoined
            })
            return rejoined
        return full_match

    # Matches single letters separated by single space (length >= 3 letters)
    pattern = re.compile(r"\b[a-zA-Z](?:\s+[a-zA-Z]){2,}\b")
    normalized = pattern.sub(repl, text)
    return normalized, mods


def normalize_delimited_words(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Reassembles words separated by decorative punctuation.
    Example: 'w*i*n*n*e*r' -> 'winner', 'f.r.e.e' -> 'free', 'w-i-n-n-e-r' -> 'winner'.
    """
    mods = []

    def repl(match):
        full_match = match.group(0)
        letters = re.split(r"[*._\-~|/]", full_match)
        rejoined = "".join(letters)
        if len(rejoined) >= 3 and rejoined.isalpha():
            mods.append({
                "type": "Delimited Word Reassembly",
                "original": full_match,
                "normalized": rejoined
            })
            return rejoined
        return full_match

    pattern = re.compile(r"\b[a-zA-Z](?:[*._\-~|/][a-zA-Z]){2,}\b")
    normalized = pattern.sub(repl, text)
    return normalized, mods


def normalize_token_leetspeak_and_abbrev(token: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Normalizes a single token for leetspeak numbers, symbol substitutions, and SMS slang.
    Preserves protected numbers, currency, and capitalization.
    """
    mods = []
    lead, core, trail = _split_affixes(token)

    if not core:
        return token, mods

    # 1. Protected entity check
    if _is_protected_numeric(core):
        return token, mods

    orig_core = core

    # 2. SMS Abbreviation lookup
    core_lower = core.lower()
    if core_lower in SMS_ABBREVIATIONS:
        expansion = SMS_ABBREVIATIONS[core_lower]
        if core.isupper():
            expansion = expansion.upper()
        elif core[0].isupper():
            expansion = expansion.capitalize()
        mods.append({
            "type": "SMS Abbreviation",
            "original": orig_core,
            "normalized": expansion
        })
        return lead + expansion + trail, mods

    # 3. Leetspeak Number and Symbol Substitution
    # Only applies if word has alphabetic characters AND leetspeak digits/symbols
    has_alpha = bool(re.search(r"[a-zA-Z]", core))
    has_leet = bool(re.search(r"[0-9@$!€£]", core))

    if has_alpha and has_leet:
        is_upper = core.isupper()
        reconstructed = []
        modified = False

        for ch in core:
            if ch in LEET_NUMBER_MAP:
                rep = LEET_NUMBER_MAP[ch]
                reconstructed.append(rep.upper() if is_upper else rep)
                modified = True
            elif ch in SYMBOL_SUB_MAP:
                rep = SYMBOL_SUB_MAP[ch]
                reconstructed.append(rep.upper() if is_upper else rep)
                modified = True
            else:
                reconstructed.append(ch)

        if modified:
            new_core = "".join(reconstructed)
            mods.append({
                "type": "Leetspeak / Symbol Substitution",
                "original": orig_core,
                "normalized": new_core
            })
            core = new_core

    return lead + core + trail, mods


def normalize_sms(text: str) -> Dict[str, Any]:
    """
    Executes the complete SMS Normalization suite on raw text:
    1. Spaced-out word reassembly
    2. Delimited word reassembly
    3. Repeated character reduction
    4. Leetspeak, symbol, and SMS abbreviation normalization

    Parameters
    ----------
    text : str
        The input SMS message string.

    Returns
    -------
    dict
        Dictionary containing:
        - 'original_text': str
        - 'normalized_text': str
        - 'modifications': list of applied transformations
        - 'has_modifications': bool
        - 'modification_count': int
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "original_text": text or "",
            "normalized_text": text or "",
            "modifications": [],
            "has_modifications": False,
            "modification_count": 0,
        }

    all_mods: List[Dict[str, str]] = []

    # Step 1: Reassemble spaced-out words (e.g. 'w i n n e r' -> 'winner')
    current_text, spaced_mods = normalize_spaced_words(text)
    all_mods.extend(spaced_mods)

    # Step 2: Reassemble delimited words (e.g. 'w*i*n*n*e*r' -> 'winner')
    current_text, delim_mods = normalize_delimited_words(current_text)
    all_mods.extend(delim_mods)

    # Step 3: Condense elongated characters (e.g. 'freeeee' -> 'free')
    current_text, repeat_mods = normalize_repeated_characters(current_text)
    all_mods.extend(repeat_mods)

    # Step 4: Token-level leetspeak and abbreviation normalization
    tokens = current_text.split()
    normalized_tokens = []
    for token in tokens:
        norm_t, token_mods = normalize_token_leetspeak_and_abbrev(token)
        normalized_tokens.append(norm_t)
        all_mods.extend(token_mods)

    final_text = " ".join(normalized_tokens)

    return {
        "original_text": text,
        "normalized_text": final_text,
        "modifications": all_mods,
        "has_modifications": len(all_mods) > 0,
        "modification_count": len(all_mods),
    }

