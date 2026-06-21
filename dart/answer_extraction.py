"""Answer extraction and normalization for SC-Route.

Extracts final answers from model output and normalizes them for
equivalence comparison across multiple drafts.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AnswerExtractor:
    """Extract and normalize answers from model output text."""

    def extract(self, text: str, task_type: str) -> str:
        if task_type == "math":
            return self._extract_math(text)
        elif task_type == "mcq":
            return self._extract_mcq(text)
        elif task_type == "code":
            return self._extract_code(text)
        return text.strip()

    def normalize(self, answer: str, task_type: str) -> str:
        if task_type == "math":
            return self._normalize_math(answer)
        elif task_type == "mcq":
            return answer.strip().upper()
        elif task_type == "code":
            return answer.strip()
        return answer.strip().lower()

    def extract_and_normalize(self, text: str, task_type: str) -> str:
        raw = self.extract(text, task_type)
        return self.normalize(raw, task_type)

    def equivalent(self, a1: str, a2: str, task_type: str) -> bool:
        n1 = self.normalize(a1, task_type) if a1 else ""
        n2 = self.normalize(a2, task_type) if a2 else ""
        if n1 == n2:
            return True
        if task_type == "math":
            return self._numeric_equivalent(n1, n2)
        return False

    # ── Math extraction ──

    def _extract_math(self, text: str) -> str:
        # Priority 1: \boxed{...} (handles nested braces)
        boxed = self._extract_boxed(text)
        if boxed is not None:
            return boxed.strip()

        # Priority 2: #### answer (GSM8K format)
        hash_match = re.search(r'####\s*(.+)', text)
        if hash_match:
            return hash_match.group(1).strip().replace(',', '')

        # Priority 3: "answer is X" pattern
        ans_match = re.search(
            r'(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(.+?)(?:\.|$)',
            text, re.IGNORECASE
        )
        if ans_match:
            return ans_match.group(1).strip()

        # Priority 4: last number in text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]

        return text.strip()[-50:] if text.strip() else ""

    def _normalize_math(self, answer: str) -> str:
        answer = answer.strip()

        # Handle \frac{a}{b} and \dfrac{a}{b} → a/b BEFORE stripping braces
        answer = re.sub(r'\\d?frac\{([^{}]+)\}\{([^{}]+)\}', r'\1/\2', answer)

        # Handle \sqrt{x} → sqrt(x)
        answer = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', answer)

        # Remove LaTeX commands (but not content)
        answer = re.sub(r'\\(?:text|mathrm|left|right|boxed|displaystyle|dfrac|tfrac|operatorname)\s*', '', answer)
        answer = re.sub(r'[{}\\$]', '', answer)
        answer = answer.replace(',', '').replace(' ', '')

        # Handle pi
        answer = re.sub(r'(\d)pi', r'\1*3.141592653589793', answer)
        answer = re.sub(r'^pi$', '3.141592653589793', answer)

        # Try numeric conversion
        try:
            val = float(answer)
            if val == int(val) and abs(val) < 1e15:
                return str(int(val))
            return f"{val:.10g}"
        except (ValueError, TypeError):
            pass

        # Try fraction pattern a/b
        frac_match = re.match(r'^(-?[\d.]+)/([\d.]+)$', answer)
        if frac_match:
            try:
                num, den = float(frac_match.group(1)), float(frac_match.group(2))
                if den != 0:
                    val = num / den
                    if val == int(val) and abs(val) < 1e15:
                        return str(int(val))
                    return f"{val:.10g}"
            except (ValueError, TypeError):
                pass

        # Try simple arithmetic: a+b, a-b, a*b
        arith_match = re.match(r'^(-?[\d.]+)([+\-*/])(-?[\d.]+)$', answer)
        if arith_match:
            try:
                a, op, b = float(arith_match.group(1)), arith_match.group(2), float(arith_match.group(3))
                if op == '+': val = a + b
                elif op == '-': val = a - b
                elif op == '*': val = a * b
                elif op == '/' and b != 0: val = a / b
                else: val = None
                if val is not None:
                    if val == int(val) and abs(val) < 1e15:
                        return str(int(val))
                    return f"{val:.10g}"
            except (ValueError, TypeError):
                pass

        return answer.strip().lower()

    def _numeric_equivalent(self, a: str, b: str) -> bool:
        try:
            va, vb = float(a), float(b)
            return abs(va - vb) < 1e-6
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _extract_boxed(text: str) -> Optional[str]:
        """Extract content from \\boxed{...} handling nested braces."""
        # Find all \boxed occurrences and return the last one
        result = None
        for match in re.finditer(r'\\boxed\{', text):
            start = match.end()
            depth = 1
            i = start
            while i < len(text) and depth > 0:
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                i += 1
            if depth == 0:
                result = text[start:i - 1]
        return result

    # ── MCQ extraction ──

    def _extract_mcq(self, text: str) -> str:
        # "answer is (A)" or "answer: B"
        match = re.search(
            r'(?:answer|choice)\s*(?:is|:)\s*\(?([A-Ja-j])\)?',
            text, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        # \boxed{A}
        boxed = re.findall(r'\\boxed\{([A-Ja-j])\}', text)
        if boxed:
            return boxed[-1].upper()

        # "therefore A"
        match = re.search(
            r'(?:therefore|so|thus|hence)\s*,?\s*\(?([A-Ja-j])\)?',
            text, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        # Last standalone letter
        match = re.search(r'\b([A-Ja-j])\s*[.)\]]?\s*$', text.strip())
        if match:
            return match.group(1).upper()

        return text.strip()[:1].upper() if text.strip() else ""

    # ── Code extraction ──

    def _extract_code(self, text: str) -> str:
        code_match = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return text.strip()
