import re
import pickle
from typing import Any

class PersianNormalizer:
    """
    Minimal Persian normalizer.

    - Unifies Arabic/Persian letter variants (ي -> ی, ك -> ک)
    - Removes Arabic diacritics and zero-width joiners
    - Optionally removes punctuation (default True)
    - Keeps digits and latin letters (important for model numbers)
    """
    def __init__(self, remove_punct: bool = True):
        self.remove_punct = remove_punct
        # unicode mappings: Arabic forms -> Persian forms
        self.mapping = {
            '\u064a': '\u06CC',  # Arabic Yeh -> Persian Yeh
            '\u0643': '\u06A9',  # Arabic Kaf -> Persian Kaf
        }
        # diacritics ranges used in Arabic script
        self._diacritics_re = re.compile(r'[\u064B-\u065F\u0610-\u061A\u06D6-\u06ED]')

    def normalize(self, text: str) -> str:
        """Normalize a string (return '' for None)."""
        if text is None:
            return ""
        txt = str(text)
        # map variant characters
        for k, v in self.mapping.items():
            txt = txt.replace(k, v)
        # remove diacritics
        txt = self._diacritics_re.sub(' ', txt)
        # remove zero-width joiners & non-joiners
        txt = txt.replace('\u200c', ' ').replace('\u200d', ' ')
        # lowercase (affects latin too)
        txt = txt.lower()
        if self.remove_punct:
            # keep letters, digits and Arabic/Persian letters
            txt = re.sub(r"[^\w\s\u0600-\u06FF]", ' ', txt)
        # collapse whitespace
        txt = re.sub(r'\s+', ' ', txt).strip()
        return txt

# simple pickle helpers (module-level)
def save_pickle(obj: Any, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)
