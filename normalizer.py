# import re
# import pickle
# from typing import Any

# class PersianNormalizer:
#     """
#     Minimal Persian normalizer.

#     - Unifies Arabic/Persian letter variants (ي -> ی, ك -> ک)
#     - Removes Arabic diacritics and zero-width joiners
#     - Optionally removes punctuation (default True)
#     - Keeps digits and latin letters (important for model numbers)
#     """
#     def __init__(self, remove_punct: bool = True):
#         self.remove_punct = remove_punct
#         # unicode mappings: Arabic forms -> Persian forms
#         self.mapping = {
#             '\u064a': '\u06CC',  # Arabic Yeh -> Persian Yeh
#             '\u0643': '\u06A9',  # Arabic Kaf -> Persian Kaf
#         }
#         # diacritics ranges used in Arabic script
#         self._diacritics_re = re.compile(r'[\u064B-\u065F\u0610-\u061A\u06D6-\u06ED]')

#     def normalize(self, text: str) -> str:
#         """Normalize a string (return '' for None)."""
#         if text is None:
#             return ""
#         txt = str(text)
#         # map variant characters
#         for k, v in self.mapping.items():
#             txt = txt.replace(k, v)
#         # remove diacritics
#         txt = self._diacritics_re.sub(' ', txt)
#         # remove zero-width joiners & non-joiners
#         txt = txt.replace('\u200c', ' ').replace('\u200d', ' ')
#         # lowercase (affects latin too)
#         txt = txt.lower()
#         if self.remove_punct:
#             # keep letters, digits and Arabic/Persian letters
#             txt = re.sub(r"[^\w\s\u0600-\u06FF]", ' ', txt)
#         # collapse whitespace
#         txt = re.sub(r'\s+', ' ', txt).strip()
#         return txt

# # simple pickle helpers (module-level)
# def save_pickle(obj: Any, path: str):
#     with open(path, 'wb') as f:
#         pickle.dump(obj, f)

# def load_pickle(path: str) -> Any:
#     with open(path, 'rb') as f:
#         return pickle.load(f)



import re
import pickle
import unicodedata
from typing import Any, Optional

# --- Precompiled regexes ---
# Arabic diacritics / tashkeel ranges + other marks
_DIACRITICS_RE = re.compile(
    r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED\u08D4-\u08E1]'
)

# tatweel (kashida)
_TATWEEL_RE = re.compile(r'\u0640')

# punctuation characters (we'll use this to remove punctuation, but allow whitelist)
_PUNCT_RE = re.compile(r'[^\w\s\u0600-\u06FF]')

# keep ZWNJ & ZWJ as chars to remove completely (do not replace with space)
_ZW_RE = re.compile(r'[\u200c\u200d]')

# NFC normalizer
def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


class PersianNormalizer:
    """
    Improved Persian/Arabic normalizer.

    Args:
        remove_punct (bool): Whether to remove punctuation. Default True.
        keep_punct (str|None): string of punctuation characters to keep even if remove_punct=True,
                               e.g. "-/._" to preserve hyphen, slash, dot, underscore in product codes.
        map_chars (bool): whether to apply Arabic->Persian character mappings (default True).
        map_digits (bool): If True, does NOT map digits here (keep digit mapping in preprocessor).
                           Set to True/False depending on where you prefer digit normalization.
    """

    # mapping table for single-character canonicalization (Arabic forms -> Persian forms)
    _CHAR_MAP = {
        # alef variants -> alef
        '\u0622': '\u0627',  # آ -> ا
        '\u0623': '\u0627',  #أ -> ا
        '\u0625': '\u0627',  #إ -> ا
        # yeh variants -> Persian ye
        '\u0649': '\u06CC',  # ى -> ی
        '\u064A': '\u06CC',  # ي -> ی
        # kaf variants -> Persian kaf
        '\u0643': '\u06A9',  # ك -> ک
        # hamza variants -> remove hamza or map to simpler forms
        '\u0624': '\u0648',  # ؤ -> و
        '\u0626': '\u06CC',  # ئ -> ی (approx)
        '\u0621': '',        # ء -> remove
        # other useful normalizations
        '\u06C0': '\u06C0',  # keep Heh + Yeh above as-is (placeholder)
        # tatweel will be removed separately
    }

    def __init__(self,
                 remove_punct: bool = True,
                 keep_punct: Optional[str] = None,
                 map_chars: bool = True,
                 map_digits: bool = False):
        self.remove_punct = remove_punct
        self.keep_punct = keep_punct or ""
        self.map_chars = map_chars
        self.map_digits = map_digits
        # build a translation table (ord -> ord or ord -> str)
        trans = {}
        for k, v in self._CHAR_MAP.items():
            if v == "":
                trans[ord(k)] = None  # drop
            else:
                trans[ord(k)] = v
        # also remove tatweel
        trans[0x0640] = None
        # store table
        self._trans_table = trans

    def _apply_char_map(self, txt: str) -> str:
        # use translate with our mapping table for speed
        return txt.translate(self._trans_table)

    def normalize(self, text: Optional[str]) -> str:
        """
        Normalize a string. Returns empty string for None input.

        Steps:
         - NFC normalize
         - apply character canonicalization (Alef/Yeh/Kaf/Hamza)
         - remove diacritics/tatweel
         - remove zero-width joiners/non-joiners (completely)
         - optional punctuation removal (with keep list)
         - casefold (for Latin)
         - collapse whitespace
        """
        if text is None:
            return ""

        txt = str(text)
        txt = _nfc(txt)

        # basic mapping of variants
        if self.map_chars:
            txt = self._apply_char_map(txt)

        # remove diacritics (tashkeel)
        txt = _DIACRITICS_RE.sub('', txt)

        # remove tatweel (kashida) if present (already mapped/trans in table, but safe)
        txt = _TATWEEL_RE.sub('', txt)

        # remove zero-width chars entirely (do NOT replace with space)
        txt = _ZW_RE.sub('', txt)

        # optionally remove punctuation but allow keep_punct chars
        if self.remove_punct:
            if self.keep_punct:
                # build temporary regex that removes punctuation except chars in keep_punct
                # escape keep_punct for regex safety
                esc = re.escape(self.keep_punct)
                # remove everything that's not word/space/arabic-range OR in keep list
                pat = re.compile(rf"(?![{esc}])[^\w\s\u0600-\u06FF]")
                txt = pat.sub(' ', txt)
            else:
                txt = _PUNCT_RE.sub(' ', txt)

        # casefold for better Unicode case handling
        txt = txt.casefold()

        # optional digit mapping could be done here if desired (we prefer pipeline-level mapping)
        if self.map_digits:
            # map Persian/Arabic digits to ASCII (simple approach)
            txt = txt.translate(str.maketrans({
                '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
                '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
                '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
                '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
            }))

        # collapse whitespace
        txt = re.sub(r'\s+', ' ', txt).strip()
        return txt


# pickle helpers
def save_pickle(obj: Any, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)
