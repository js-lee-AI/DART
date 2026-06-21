import re
import logging
from typing import Optional
logger = logging.getLogger(__name__)

class AnswerExtractor:

    def extract(self, text: str, task_type: str) -> str:
        if task_type == 'math':
            return self._extract_math(text)
        elif task_type == 'mcq':
            return self._extract_mcq(text)
        elif task_type == 'code':
            return self._extract_code(text)
        return text.strip()

    def normalize(self, answer: str, task_type: str) -> str:
        if task_type == 'math':
            return self._normalize_math(answer)
        elif task_type == 'mcq':
            return answer.strip().upper()
        elif task_type == 'code':
            return answer.strip()
        return answer.strip().lower()

    def extract_and_normalize(self, text: str, task_type: str) -> str:
        raw = self.extract(text, task_type)
        return self.normalize(raw, task_type)

    def equivalent(self, a1: str, a2: str, task_type: str) -> bool:
        n1 = self.normalize(a1, task_type) if a1 else ''
        n2 = self.normalize(a2, task_type) if a2 else ''
        if n1 == n2:
            return True
        if task_type == 'math':
            return self._numeric_equivalent(n1, n2)
        return False

    def _extract_math(self, text: str) -> str:
        boxed = self._extract_boxed(text)
        if boxed is not None:
            return boxed.strip()
        hash_match = re.search('####\\s*(.+)', text)
        if hash_match:
            return hash_match.group(1).strip().replace(',', '')
        ans_match = re.search('(?:the\\s+)?(?:final\\s+)?answer\\s+is\\s*:?\\s*(.+?)(?:\\.|$)', text, re.IGNORECASE)
        if ans_match:
            return ans_match.group(1).strip()
        numbers = re.findall('-?\\d+\\.?\\d*', text)
        if numbers:
            return numbers[-1]
        return text.strip()[-50:] if text.strip() else ''

    def _normalize_math(self, answer: str) -> str:
        answer = answer.strip()
        answer = re.sub('\\\\d?frac\\{([^{}]+)\\}\\{([^{}]+)\\}', '\\1/\\2', answer)
        answer = re.sub('\\\\sqrt\\{([^{}]+)\\}', 'sqrt(\\1)', answer)
        answer = re.sub('\\\\(?:text|mathrm|left|right|boxed|displaystyle|dfrac|tfrac|operatorname)\\s*', '', answer)
        answer = re.sub('[{}\\\\$]', '', answer)
        answer = answer.replace(',', '').replace(' ', '')
        answer = re.sub('(\\d)pi', '\\1*3.141592653589793', answer)
        answer = re.sub('^pi$', '3.141592653589793', answer)
        try:
            val = float(answer)
            if val == int(val) and abs(val) < 1000000000000000.0:
                return str(int(val))
            return f'{val:.10g}'
        except (ValueError, TypeError):
            pass
        frac_match = re.match('^(-?[\\d.]+)/([\\d.]+)$', answer)
        if frac_match:
            try:
                (num, den) = (float(frac_match.group(1)), float(frac_match.group(2)))
                if den != 0:
                    val = num / den
                    if val == int(val) and abs(val) < 1000000000000000.0:
                        return str(int(val))
                    return f'{val:.10g}'
            except (ValueError, TypeError):
                pass
        arith_match = re.match('^(-?[\\d.]+)([+\\-*/])(-?[\\d.]+)$', answer)
        if arith_match:
            try:
                (a, op, b) = (float(arith_match.group(1)), arith_match.group(2), float(arith_match.group(3)))
                if op == '+':
                    val = a + b
                elif op == '-':
                    val = a - b
                elif op == '*':
                    val = a * b
                elif op == '/' and b != 0:
                    val = a / b
                else:
                    val = None
                if val is not None:
                    if val == int(val) and abs(val) < 1000000000000000.0:
                        return str(int(val))
                    return f'{val:.10g}'
            except (ValueError, TypeError):
                pass
        return answer.strip().lower()

    def _numeric_equivalent(self, a: str, b: str) -> bool:
        try:
            (va, vb) = (float(a), float(b))
            return abs(va - vb) < 1e-06
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _extract_boxed(text: str) -> Optional[str]:
        result = None
        for match in re.finditer('\\\\boxed\\{', text):
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

    def _extract_mcq(self, text: str) -> str:
        match = re.search('(?:answer|choice)\\s*(?:is|:)\\s*\\(?([A-Ja-j])\\)?', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        boxed = re.findall('\\\\boxed\\{([A-Ja-j])\\}', text)
        if boxed:
            return boxed[-1].upper()
        match = re.search('(?:therefore|so|thus|hence)\\s*,?\\s*\\(?([A-Ja-j])\\)?', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        match = re.search('\\b([A-Ja-j])\\s*[.)\\]]?\\s*$', text.strip())
        if match:
            return match.group(1).upper()
        return text.strip()[:1].upper() if text.strip() else ''

    def _extract_code(self, text: str) -> str:
        code_match = re.search('```(?:python)?\\n(.*?)```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return text.strip()
