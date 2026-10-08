"""Rottweiler Verification: 5-Stage XML Scaffold and Sandboxed SymPy Verifier (Requirement R5).

Enforces:
1. 5-stage XML scaffold (<explore>, <conjecture>, <test_edge_cases>, <lemma_isolate>, <formal_proof>)
   with strict sequence, zero nesting, non-empty blocks, and terminal \\boxed{answer} inside <formal_proof>.
2. AST-sandboxed SymPy execution verifier testing symbolic equivalence and reverse equation substitution.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import ast
import math
import re
import signal
import sys
import threading
import sympy


TAG_NAMES = [
    "explore",
    "conjecture",
    "test_edge_cases",
    "lemma_isolate",
    "formal_proof"
]

SAFE_AST_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.USub, ast.UAdd,
    ast.Name, ast.Call, ast.Tuple, ast.List, ast.Load, ast.Compare, ast.Eq
}

ALLOWED_MATH_FUNCS = {
    "sqrt", "pi", "sin", "cos", "tan", "log", "exp", "abs",
    "Rational", "Integer", "Matrix", "frac", "Symbol", "factorial", "oo"
}

BANNED_NAMES = {
    "eval", "exec", "open", "__import__", "compile", "globals", "locals",
    "getattr", "setattr", "delattr", "hasattr", "input", "print", "breakpoint",
    "help", "exit", "quit", "system", "popen", "spawn", "os", "sys", "subprocess", "pathlib",
    "shutil", "time", "socket", "urllib", "requests"
}


class TimeoutError(Exception):
    """Raised when symbolic verification exceeds execution timeout."""
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


def run_with_timeout(fn, args=(), kwargs=None, timeout_seconds: float = 5.0):
    """Executes a function with POSIX signal alarm or direct call."""
    if kwargs is None:
        kwargs = {}

    is_main_thread = (threading.current_thread() is threading.main_thread())
    if is_main_thread and hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(math.ceil(timeout_seconds)))
        try:
            return fn(*args, **kwargs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Fallback in non-main thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn, *args, **kwargs)
            return future.result(timeout=timeout_seconds)


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extracts contents of terminal \\boxed{...} handling nested curly braces."""
    patterns = [r"\boxed{", "boxed{"]
    idx = -1
    lead_len = 0
    for pat in patterns:
        cur_idx = text.rfind(pat)
        if cur_idx > idx:
            idx = cur_idx
            lead_len = len(pat)

    if idx == -1:
        return None

    idx += lead_len
    depth = 1
    chars = []
    for c in text[idx:]:
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                break
        chars.append(c)

    if depth == 0:
        return "".join(chars).strip()
    return None


def clean_latex_math(expr: str, is_equation: bool = False) -> str:
    """Sanitizes LaTeX mathematical expressions for SymPy evaluation."""
    s = expr.strip()
    s = s.replace("$", "")
    s = s.replace(r"\(", "").replace(r"\)", "")
    s = s.replace(r"\[", "").replace(r"\]", "")

    # Convert exponentiation ^ to Python **
    s = s.replace("^", "**")

    # If expression is written as variable assignment (e.g., "x = 42") and is NOT an equation to verify, isolate RHS
    if not is_equation:
        m_eq = re.match(r"^[a-zA-Z]\s*=\s*(.+)$", s)
        if m_eq:
            s = m_eq.group(1).strip()

    # Normalize nested fractions: \frac{num}{den} and \dfrac{num}{den} -> ((num)/(den))
    for frac_cmd in [r"\frac{", r"\dfrac{"]:
        while frac_cmd in s:
            idx = s.find(frac_cmd)
            start_num = idx + len(frac_cmd) - 1
            depth = 0
            end_num = -1
            for i in range(start_num, len(s)):
                if s[i] == "{":
                    depth += 1
                elif s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end_num = i
                        break
            if end_num == -1:
                break

            # Handle possible whitespace between {num} and {den} in LaTeX
            rest = s[end_num + 1:]
            lstripped = rest.lstrip()
            if not lstripped.startswith("{"):
                break
            start_den = end_num + 1 + (len(rest) - len(lstripped))
            depth = 0
            end_den = -1
            for i in range(start_den, len(s)):
                if s[i] == "{":
                    depth += 1
                elif s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end_den = i
                        break
            if end_den == -1:
                break
            num = s[start_num + 1:end_num]
            den = s[start_den + 1:end_den]
            s = s[:idx] + f"(({num})/({den}))" + s[end_den + 1:]

    # Normalize nested square roots: \sqrt{arg} -> sqrt(arg)
    while r"\sqrt{" in s:
        idx = s.find(r"\sqrt{")
        depth = 0
        end_idx = -1
        for i in range(idx + 5, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        if end_idx != -1:
            arg = s[idx + 6:end_idx]
            s = s[:idx] + f"sqrt({arg})" + s[end_idx + 1:]
        else:
            break

    s = s.replace(r"\pi", "pi")
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    s = re.sub(r"\\[,;! ]", "", s)
    s = s.replace(r"\left", "").replace(r"\right", "")

    # Normalize LaTeX mathematical functions
    for fn in [
        "arcsin", "arccos", "arctan",
        "sinh", "cosh", "tanh",
        "sin", "cos", "tan", "sec", "csc", "cot",
        "exp", "abs", "det"
    ]:
        s = s.replace(f"\\{fn}", fn)
    s = s.replace(r"\ln", "log")
    s = s.replace(r"\log", "log")

    # Resolve implicit multiplication
    s = re.sub(r"(\d+)\s*([a-zA-Z\(])", r"\1*\2", s)
    s = re.sub(r"(\))\s*(\()", r"\1*\2", s)
    s = re.sub(r"(\))\s*([a-zA-Z0-9])", r"\1*\2", s)

    return s.strip()


def is_safe_math_expression(expr_str: str) -> bool:
    """Verifies that an expression contains only safe mathematical syntax to prevent RCE."""
    # Split equations if present for checking
    if "=" in expr_str:
        parts = expr_str.split("=")
        return all(is_safe_math_expression(part.strip()) for part in parts if part.strip())

    try:
        tree = ast.parse(expr_str, mode="eval")
    except Exception:
        return False

    for node in ast.walk(tree):
        if type(node) not in SAFE_AST_NODES:
            return False
        if isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in BANNED_NAMES:
                return False
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return False
            if node.func.id not in ALLOWED_MATH_FUNCS:
                return False

    return True


class XMLScaffoldValidator:
    """Validates 5-stage XML reasoning scaffold."""

    TAGS = TAG_NAMES

    @classmethod
    def extract_tags(cls, text: str) -> Dict[str, Optional[str]]:
        """Extracts text content within each tag."""
        extracted = {}
        for tag in cls.TAGS:
            open_marker = f"<{tag}>"
            close_marker = f"</{tag}>"
            if open_marker in text and close_marker in text:
                start = text.find(open_marker) + len(open_marker)
                end = text.find(close_marker)
                if start <= end:
                    extracted[tag] = text[start:end].strip()
                else:
                    extracted[tag] = None
            else:
                extracted[tag] = None
        return extracted

    @classmethod
    def validate(cls, text: str) -> Dict[str, Any]:
        """Validates scaffold structure, tag ordering, nesting, and boxed answer placement.

        Returns:
            Dict containing detailed adherence report.
        """
        tags_present: Dict[str, bool] = {}
        error_messages: List[str] = []

        # 1. Detect corrupted tags and stray punctuation attached to tag boundaries
        for tag in cls.TAGS:
            # Corrupted closing tag prefix (e.g., *</explore>, :</explore>, **</explore>)
            prefix_close_pat = r"\s+[:*#`~_\-;]+\s*</" + tag + ">"
            direct_close_pat = r"[\*\:\#\`\~_\-;]\s*</" + tag + ">"
            if re.search(prefix_close_pat, text) or re.search(direct_close_pat, text):
                error_messages.append(f"Corrupted closing tag detected: stray prefix attached to </{tag}>")

            # Corrupted closing tag suffix (e.g., </explore>*, </explore> :)
            suffix_close_pat = r"</" + tag + r">\s*[\*:\#\`\~_\-;]+"
            if re.search(suffix_close_pat, text):
                error_messages.append(f"Corrupted closing tag detected: stray suffix attached to </{tag}>")

            # Corrupted opening tag prefix (e.g., *<explore>, :<explore>)
            prefix_open_pat = r"[\*\:\#\`\~_\-;]\s*<" + tag + ">"
            prefix_space_open_pat = r"\s+[:*#`~_\-;]+\s*<" + tag + ">"
            if re.search(prefix_open_pat, text) or re.search(prefix_space_open_pat, text):
                error_messages.append(f"Corrupted opening tag detected: stray prefix attached to <{tag}>")

            # Corrupted opening tag suffix (e.g., <explore>*, <explore> :)
            suffix_open_pat = r"<" + tag + r">\s*[\*:\#\`\~_\-;]+(?:\s*[\r\n]|\s*<)"
            if re.search(suffix_open_pat, text):
                error_messages.append(f"Corrupted opening tag detected: stray suffix attached to <{tag}>")

            # Malformed tag syntax (whitespace or punctuation inside tag e.g. < /explore>, <explore >, <*explore>, </*explore>)
            malformed_tag_pat = r"<[\s/\\*:#`~_\-;]*" + tag + r"[\s/\\*:#`~_\-;]*>"
            for m in re.finditer(malformed_tag_pat, text, re.IGNORECASE):
                raw_tag = m.group(0)
                if raw_tag not in (f"<{tag}>", f"</{tag}>"):
                    error_messages.append(f"Malformed tag syntax detected: '{raw_tag}'")

        # 2. Exact count of open and close tags
        for tag in cls.TAGS:
            o_cnt = text.count(f"<{tag}>")
            c_cnt = text.count(f"</{tag}>")
            tags_present[tag] = (o_cnt == 1 and c_cnt == 1)
            if o_cnt != 1 or c_cnt != 1:
                error_messages.append(f"Tag <{tag}> count mismatch (open={o_cnt}, close={c_cnt})")

        all_tags_present = all(tags_present.values())
        if not all_tags_present or error_messages:
            return {
                "adherent": False,
                "all_tags_present": all_tags_present,
                "strictly_ordered": False,
                "zero_nesting": False,
                "boxed_in_formal_proof": False,
                "no_trailing_text": False,
                "no_corrupted_tags": False if error_messages else True,
                "extracted_tags": cls.extract_tags(text),
                "boxed_answer": extract_boxed_answer(text),
                "error_message": "; ".join(error_messages) if error_messages else "Tags missing or corrupted",
                "format_penalty": -1.5,
            }

        # 3. Interleaved, duplicate, and strict sequential ordering validation
        expected_sequence = []
        for t in cls.TAGS:
            expected_sequence.append(f"<{t}>")
            expected_sequence.append(f"</{t}>")

        tag_pattern = re.compile(rf"</?(?:{'|'.join(cls.TAGS)})>")
        observed_tags = [m.group(0) for m in tag_pattern.finditer(text)]
        if observed_tags != expected_sequence:
            error_messages.append(f"Interleaved or out-of-order tag sequence: observed {observed_tags}")

        strictly_ordered = (observed_tags == expected_sequence)
        zero_nesting = True
        last_close_pos = -1
        extracted_tags: Dict[str, str] = {}

        for tag in cls.TAGS:
            open_pos = text.find(f"<{tag}>")
            close_pos = text.find(f"</{tag}>")

            if not (last_close_pos <= open_pos < close_pos):
                strictly_ordered = False
                error_messages.append(f"Out of order sequence around <{tag}>")
                break

            content = text[open_pos + len(f"<{tag}>"):close_pos]
            if len(content.strip()) < 5:
                error_messages.append(f"Block <{tag}> is empty or non-substantive")

            # Check for any other tag inside content
            for other_tag in cls.TAGS:
                if f"<{other_tag}>" in content or f"</{other_tag}>" in content:
                    zero_nesting = False
                    error_messages.append(f"Nesting detected inside <{tag}> with <{other_tag}>")
                    break

            extracted_tags[tag] = content.strip()
            last_close_pos = close_pos + len(f"</{tag}>")

        # 4. Terminal boxed answer inside formal_proof, and NOT outside
        formal_proof_content = extracted_tags.get("formal_proof", "")
        boxed_inside_formal = extract_boxed_answer(formal_proof_content) is not None
        if not boxed_inside_formal:
            error_messages.append("No \\boxed{...} found inside <formal_proof>")

        # Check that boxed answer does not appear before formal_proof
        formal_open_pos = text.find("<formal_proof>")
        if formal_open_pos != -1:
            boxed_before = extract_boxed_answer(text[:formal_open_pos])
            if boxed_before is not None:
                boxed_inside_formal = False
                error_messages.append("Premature \\boxed{...} detected outside <formal_proof>")

        # 5. No reasoning text following </formal_proof>
        trailing_text = text[last_close_pos:].strip()
        no_trailing_text = (len(trailing_text) == 0)
        if not no_trailing_text:
            error_messages.append("Trailing reasoning text found after </formal_proof>")

        is_adherent = (
            all_tags_present
            and strictly_ordered
            and zero_nesting
            and boxed_inside_formal
            and no_trailing_text
            and len(error_messages) == 0
        )

        return {
            "adherent": is_adherent,
            "all_tags_present": all_tags_present,
            "strictly_ordered": strictly_ordered,
            "zero_nesting": zero_nesting,
            "boxed_in_formal_proof": boxed_inside_formal,
            "no_trailing_text": no_trailing_text,
            "no_corrupted_tags": len(error_messages) == 0,
            "extracted_tags": extracted_tags,
            "boxed_answer": extract_boxed_answer(formal_proof_content) if boxed_inside_formal else None,
            "error_message": "; ".join(error_messages) if error_messages else None,
            "format_penalty": 0.0 if is_adherent else -1.5,
        }


class LemmaConsistencyValidator:
    """Validates theorem preconditions and semantic concordance between <lemma_isolate> and <formal_proof>."""

    THEOREM_KEYWORDS = {
        "chinese_remainder": ["congruen", "modulo", "mod ", "coprime", "crt", "remainder", "system"],
        "wilson": ["wilson", "(p-1)!", "(n-1)!", "factorial", "!", "residue", "mod "],
        "fermat_little": ["fermat", "p-1", "mod ", "totient", "power", "prime"],
        "euler_totient": ["euler", "totient", "phi", "\\phi", "coprime", "mod "],
        "legendre_formula": ["legendre", "v_p", "v_", "highest power", "factorial", "\\lfloor", "floor"],
        "divisor_multiplicity": ["divisor", "d(n)", "tau", "factor", "multiplicity", "\\prod", "divisible", "divides"],
        "euclidean_algorithm": ["gcd", "euclid", "divisor", "remainder", "mod "],
        "pythagorean": ["pythagor", "right triangle", "hypotenuse", "leg", "a^2 + b^2", "triangle"],
        "heron": ["heron", "semi-perimeter", "area", "triangle", "\\sqrt{s"],
        "angle_bisector": ["bisector", "ratio", "angle", "triangle"],
        "power_of_a_point": ["power of a point", "chord", "secant", "tangent", "circle"],
        "inradius_area": ["inradius", "incircle", "semi-perimeter", "rs", "area"],
        "stars_and_bars": ["stars and bars", "indistinguishable", "bin", "distribute", "\\binom", "choose"],
        "inclusion_exclusion": ["inclusion-exclusion", "pie", "union", "intersect", "double-count", "complement"],
        "vieta": ["vieta", "sum of roots", "product of roots", "root", "coefficient", "polynomial"],
        "difference_of_squares": ["difference of squares", "factor", "a^2 - b^2", "complete the square", "quadratic"],
        "am_gm": ["am-gm", "arithmetic mean", "geometric mean", "inequality", "equality holds"],
        "cauchy_schwarz": ["cauchy-schwarz", "inner product", "collinear", "dot product", "inequality"],
        "telescoping": ["telescop", "cancel", "partial fraction", "sum", "series"],
        "rational_root": ["rational root", "p/q", "dividing the constant", "leading coefficient"],
        "de_moivre": ["de moivre", "complex", "euler", "cis", "polar", "trigonometric", "cos", "sin"]
    }

    @classmethod
    def extract_moduli(cls, text: str) -> List[int]:
        """Extracts candidate integer moduli from text."""
        moduli = []
        patterns = [
            r"\\[pb]?mod\s*\{(?:\s*[a-zA-Z]\s*=)?\s*(\d+)\s*\}",
            r"\\[pb]?mod\s*\(?(?:\s*[a-zA-Z]\s*=)?\s*(\d+)\s*\)?",
            r"\bmod(?:ulo)?\s*(?:is|=|:)?\s*(?:[a-zA-Z]\s*=\s*)?(\d+)\b",
            r"\bmodulus\s*(?:is|=|:)?\s*(?:[a-zA-Z]\s*=\s*)?(\d+)\b",
            r"\bprime\s*(?:is|=|:)?\s*(?:[a-zA-Z]\s*=\s*)?(\d+)\b",
            r"\b[pnm]\s*=\s*(\d+)\b",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                try:
                    moduli.append(int(m.group(1)))
                except ValueError:
                    pass
        return list(set(moduli))

    @classmethod
    def get_associated_moduli(cls, text: str, theorem_terms: List[str]) -> List[int]:
        """Extracts moduli specifically tied to sentences mentioning theorem terms.
        Falls back to all moduli in text if no direct sentence association exists."""
        sentences = re.split(r"[\n\.\;]", text)
        targeted_moduli = []
        for s in sentences:
            s_lower = s.lower()
            if any(term in s_lower for term in theorem_terms):
                m_list = cls.extract_moduli(s)
                targeted_moduli.extend(m_list)

        if targeted_moduli:
            return list(set(targeted_moduli))
        return cls.extract_moduli(text)

    @classmethod
    def validate(cls, completion: str, extracted_tags: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Any]:
        """Validates lemma precondition integrity and concordance with formal proof."""
        if extracted_tags is None:
            extracted_tags = XMLScaffoldValidator.extract_tags(completion)

        lemma_text = extracted_tags.get("lemma_isolate") or ""
        proof_text = extracted_tags.get("formal_proof") or ""

        if not lemma_text or not proof_text:
            return {
                "consistent": False,
                "error_message": "Missing lemma_isolate or formal_proof block",
                "penalty": -1.5
            }

        lemma_lower = lemma_text.lower()
        proof_lower = proof_text.lower()
        combined_text = f"{lemma_text}\n{proof_text}"

        is_euler = bool(("euler" in lemma_lower and "totient" in lemma_lower) or "\\phi" in lemma_lower or "phi(" in lemma_lower)
        is_crt = bool("chinese remainder" in lemma_lower or "crt" in lemma_lower)

        # 1. Precondition Sanity Check: Wilson's Theorem
        wilson_terms = ["wilson", "(p-1)!", "(n-1)!"]
        has_modular_context_lemma = any(w in lemma_lower for w in ["mod", "equiv", "residue", "congruen", "prime", "wilson"])
        is_wilson_lemma = bool(
            (re.search(r"wilson(?:'s)?\s*(?:theorem)?", lemma_lower) or (has_modular_context_lemma and ("(p-1)!" in lemma_lower or "(n-1)!" in lemma_lower)))
            and not any(neg in lemma_lower for neg in ["not apply", "does not apply", "cannot be applied", "cannot apply", "fails", "not applicable"])
        )
        has_modular_context_proof = any(w in proof_lower for w in ["mod", "equiv", "residue", "congruen", "prime", "wilson"])
        is_wilson_proof = bool(
            (re.search(r"wilson(?:'s)?\s*(?:theorem)?", proof_lower) or (has_modular_context_proof and ("(p-1)!" in proof_lower or "(n-1)!" in proof_lower)))
            and not any(neg in proof_lower for neg in ["not apply", "does not apply", "cannot be applied", "cannot apply", "fails", "not applicable"])
        )
        is_wilson = is_wilson_lemma or is_wilson_proof
        if is_wilson:
            w_moduli = cls.get_associated_moduli(combined_text, wilson_terms)
            for m in w_moduli:
                if not sympy.isprime(m):
                    return {
                        "consistent": False,
                        "error_message": f"Invalid theorem precondition: Wilson's Theorem requires prime modulus, but modulus {m} is composite",
                        "penalty": -1.5
                    }

        # 2. Precondition Sanity Check: Fermat's Little Theorem
        flt_terms = ["fermat", "a^{p-1}", "a^{n-1}"]
        has_mod_flt_lemma = any(w in lemma_lower for w in ["mod", "equiv", "residue", "congruen", "prime", "fermat"])
        is_flt_lemma = bool(
            (re.search(r"fermat(?:'s)?\s*(?:little)?\s*theorem", lemma_lower) or (has_mod_flt_lemma and ("a^{p-1}" in lemma_lower or "a^{n-1}" in lemma_lower)))
            and not is_euler
            and not is_crt
            and not any(neg in lemma_lower for neg in ["not apply", "does not apply", "cannot be applied", "cannot apply", "fails", "not applicable"])
        )
        has_mod_flt_proof = any(w in proof_lower for w in ["mod", "equiv", "residue", "congruen", "prime", "fermat"])
        is_flt_proof = bool(
            (re.search(r"fermat(?:'s)?\s*(?:little)?\s*theorem", proof_lower) or (has_mod_flt_proof and ("a^{p-1}" in proof_lower or "a^{n-1}" in proof_lower)))
            and not is_euler
            and not is_crt
            and not any(neg in proof_lower for neg in ["not apply", "does not apply", "cannot be applied", "cannot apply", "fails", "not applicable"])
        )
        is_flt = is_flt_lemma or is_flt_proof
        if is_flt:
            flt_moduli = cls.get_associated_moduli(combined_text, flt_terms)
            for m in flt_moduli:
                if not sympy.isprime(m):
                    return {
                        "consistent": False,
                        "error_message": f"Invalid theorem precondition: Fermat's Little Theorem requires prime modulus, but modulus {m} is composite (use Euler's Totient Theorem or Chinese Remainder Theorem)",
                        "penalty": -1.5
                    }

        # 3. Semantic Concordance Check between <lemma_isolate> and <formal_proof>
        detected_classes = []
        if is_crt:
            detected_classes.append("chinese_remainder")
        if is_wilson_lemma:
            detected_classes.append("wilson")
        if is_flt_lemma:
            detected_classes.append("fermat_little")
        if is_euler:
            detected_classes.append("euler_totient")
        if "legendre" in lemma_lower or "v_p" in lemma_lower:
            detected_classes.append("legendre_formula")
        if "divisor multiplicity" in lemma_lower or "number of positive divisors" in lemma_lower:
            detected_classes.append("divisor_multiplicity")
        if re.search(r"euclid(?:ean)?(?:'s)?\s*algorithm", lemma_lower) or "gcd(a, b)" in lemma_lower:
            detected_classes.append("euclidean_algorithm")
        if "pythagor" in lemma_lower:
            detected_classes.append("pythagorean")
        if "heron" in lemma_lower:
            detected_classes.append("heron")
        if "angle bisector" in lemma_lower:
            detected_classes.append("angle_bisector")
        if "power of a point" in lemma_lower:
            detected_classes.append("power_of_a_point")
        if "inradius" in lemma_lower:
            detected_classes.append("inradius_area")
        if "stars and bars" in lemma_lower:
            detected_classes.append("stars_and_bars")
        if re.search(r"inclusion\s*[-–/]\s*exclusion", lemma_lower) or "pie" in lemma_lower:
            detected_classes.append("inclusion_exclusion")
        if "vieta" in lemma_lower:
            detected_classes.append("vieta")
        if "difference of squares" in lemma_lower or "completing the square" in lemma_lower or "a^2 - b^2" in lemma_lower:
            detected_classes.append("difference_of_squares")
        if re.search(r"am\s*[-–/]\s*gm", lemma_lower) or "arithmetic mean" in lemma_lower:
            detected_classes.append("am_gm")
        if re.search(r"cauchy\s*[-–/]\s*schwarz", lemma_lower) or "cauchy-schwarz" in lemma_lower:
            detected_classes.append("cauchy_schwarz")
        if "telescoping" in lemma_lower:
            detected_classes.append("telescoping")
        if "rational root" in lemma_lower:
            detected_classes.append("rational_root")
        if "de moivre" in lemma_lower:
            detected_classes.append("de_moivre")

        for cls_name in detected_classes:
            kw_list = cls.THEOREM_KEYWORDS.get(cls_name, [])
            if kw_list:
                found = any(kw in proof_lower for kw in kw_list)
                if not found:
                    return {
                        "consistent": False,
                        "error_message": f"Semantic divergence: declared lemma '{cls_name}' is not utilized in formal proof",
                        "penalty": -1.5
                    }

        return {
            "consistent": True,
            "error_message": None,
            "penalty": 0.0
        }


class RottweilerSymPyVerifier:
    """Sandboxed symbolic verifier checking exact equivalence and reverse substitution."""

    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout_seconds = timeout_seconds

    def clean_latex(self, expr_str: str, is_equation: bool = False) -> str:
        """Sanitizes LaTeX expression."""
        return clean_latex_math(expr_str, is_equation=is_equation)

    def is_safe_expression(self, expr_str: str) -> bool:
        """Validates AST safety to prevent code execution."""
        return is_safe_math_expression(expr_str)

    def verify_symbolic_equivalence(self, pred_raw: Optional[str], gold_raw: str) -> bool:
        """Tests mathematical equivalence between predicted answer and ground truth."""
        if not pred_raw or not gold_raw:
            return False

        pred_clean = self.clean_latex(pred_raw)
        gold_clean = self.clean_latex(gold_raw)

        # 1. Exact string match
        if pred_clean.lower() == gold_clean.lower():
            return True

        # 2. Strict numeric match for single scalars
        num_pattern = r"^[+-]?\d+(?:\.\d+)?$"
        if re.fullmatch(num_pattern, pred_clean) and re.fullmatch(num_pattern, gold_clean):
            try:
                return abs(float(pred_clean) - float(gold_clean)) < 1e-6
            except Exception:
                pass

        # 3. AST sandboxing check
        if not self.is_safe_expression(pred_clean) or not self.is_safe_expression(gold_clean):
            return False

        def _evaluate():
            try:
                sym_pred = sympy.sympify(pred_clean)
                sym_gold = sympy.sympify(gold_clean)
                diff = sympy.simplify(sym_pred - sym_gold)
                if diff == 0 or diff.is_zero:
                    return True
                if diff.is_number and abs(complex(diff.evalf())) < 1e-6:
                    return True
            except Exception:
                return False
            return False

        try:
            return run_with_timeout(_evaluate, timeout_seconds=self.timeout_seconds)
        except (TimeoutError, Exception):
            return False

    def verify_reverse_substitution(
        self,
        candidate_ans: str,
        equation_str: str,
        variable_name: str = "x",
        domain_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Substitutes candidate answer into equation and verifies residual.

        Detects extraneous roots from division-by-zero denominators and evaluates domain constraints.
        """
        cand_clean = self.clean_latex(candidate_ans, is_equation=False)
        eq_clean = self.clean_latex(equation_str, is_equation=True)

        if not self.is_safe_expression(cand_clean) or not self.is_safe_expression(eq_clean):
            return {
                "valid": False,
                "residual": None,
                "domain_valid": False,
                "error_message": "AST safety check failed for expression or candidate"
            }

        def _evaluate_sub():
            # Parse LHS and RHS
            if "=" in eq_clean:
                parts = eq_clean.split("=")
                lhs_str, rhs_str = parts[0].strip(), parts[1].strip()
            else:
                lhs_str, rhs_str = eq_clean.strip(), "0"

            var = sympy.Symbol(variable_name)
            x_val = sympy.sympify(cand_clean)
            lhs_expr = sympy.sympify(lhs_str)
            rhs_expr = sympy.sympify(rhs_str)

            # Check for division by zero / extraneous roots
            # Check powers with negative exponents or denominators
            for expr in [lhs_expr, rhs_expr]:
                for p in expr.atoms(sympy.core.power.Pow):
                    if p.exp.is_negative:
                        denom_val = sympy.simplify(p.base.subs(var, x_val))
                        if denom_val == 0 or denom_val.is_zero:
                            return {
                                "valid": False,
                                "residual": None,
                                "domain_valid": False,
                                "error_message": f"Extraneous root: denominator {p.base} evaluates to zero"
                            }

            lhs_sub = lhs_expr.subs(var, x_val)
            rhs_sub = rhs_expr.subs(var, x_val)

            # Check if substitution produced zoo or nan
            if lhs_sub.has(sympy.zoo, sympy.nan, sympy.oo) or rhs_sub.has(sympy.zoo, sympy.nan, sympy.oo):
                return {
                    "valid": False,
                    "residual": None,
                    "domain_valid": False,
                    "error_message": "Substitution produced infinite or undefined value"
                }

            # Evaluate domain constraints
            if domain_constraints:
                if domain_constraints.get("integer", False):
                    if not (x_val.is_integer or (x_val.is_number and float(x_val).is_integer())):
                        return {
                            "valid": False,
                            "residual": None,
                            "domain_valid": False,
                            "error_message": f"Candidate {cand_clean} does not satisfy integer constraint"
                        }
                if domain_constraints.get("positive", False):
                    if x_val.is_number and float(x_val) <= 0:
                        return {
                            "valid": False,
                            "residual": None,
                            "domain_valid": False,
                            "error_message": f"Candidate {cand_clean} does not satisfy positive constraint"
                        }

            # Check residual
            residual = sympy.simplify(lhs_sub - rhs_sub)
            is_zero = (residual == 0 or residual.is_zero)
            if not is_zero and residual.is_number:
                is_zero = (abs(complex(residual.evalf())) < 1e-6)

            return {
                "valid": bool(is_zero),
                "residual": str(residual),
                "domain_valid": True,
                "error_message": None if is_zero else f"Non-zero residual: {residual}"
            }

        try:
            return run_with_timeout(_evaluate_sub, timeout_seconds=self.timeout_seconds)
        except TimeoutError:
            return {
                "valid": False,
                "residual": None,
                "domain_valid": False,
                "error_message": f"Timeout of {self.timeout_seconds}s exceeded during substitution"
            }
        except Exception as e:
            return {
                "valid": False,
                "residual": None,
                "domain_valid": False,
                "error_message": str(e)
            }


class RottweilerVerifier:
    """Unified Rottweiler Verifier orchestrating XML scaffold, Lemma consistency, and SymPy verification."""

    def __init__(self, timeout_seconds: float = 5.0):
        self.scaffold_validator = XMLScaffoldValidator()
        self.lemma_validator = LemmaConsistencyValidator()
        self.sympy_verifier = RottweilerSymPyVerifier(timeout_seconds=timeout_seconds)

    def verify_xml_scaffold(self, text: str) -> Tuple[bool, str]:
        """Validates XML scaffold adherence."""
        report = self.scaffold_validator.validate(text)
        return report["adherent"], report.get("error_message") or ""

    def verify_lemma_consistency(self, completion: str) -> Tuple[bool, str, float]:
        """Validates lemma precondition integrity and concordance with formal proof."""
        report = self.lemma_validator.validate(completion)
        return report["consistent"], report.get("error_message") or "", report.get("penalty", 0.0)

    def verify_sympy_solution(
        self,
        prediction: str,
        ground_truth: str,
        equation_str: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Verifies solution equivalence and optional reverse substitution."""
        equiv = self.sympy_verifier.verify_symbolic_equivalence(prediction, ground_truth)
        if not equiv:
            return False, "Symbolic equivalence check failed"

        if equation_str:
            sub_res = self.sympy_verifier.verify_reverse_substitution(prediction, equation_str)
            if not sub_res["valid"]:
                return False, sub_res.get("error_message") or "Reverse substitution failed"

        return True, ""

    def verify(
        self,
        completion: str,
        ground_truth: str,
        equation_str: Optional[str] = None
    ) -> Dict[str, Any]:
        """Full verification pipeline returning comprehensive report."""
        scaffold_rep = self.scaffold_validator.validate(completion)
        boxed_ans = scaffold_rep.get("boxed_answer")

        if not scaffold_rep["adherent"] or not boxed_ans:
            return {
                "valid": False,
                "scaffold_adherent": scaffold_rep["adherent"],
                "lemma_consistent": False,
                "symbolic_correct": False,
                "boxed_answer": boxed_ans,
                "error": scaffold_rep.get("error_message") or "Scaffold violation",
                "format_penalty": scaffold_rep.get("format_penalty", -1.5),
                "lemma_penalty": 0.0,
            }

        lemma_rep = self.lemma_validator.validate(completion, scaffold_rep.get("extracted_tags"))
        if not lemma_rep["consistent"]:
            return {
                "valid": False,
                "scaffold_adherent": True,
                "lemma_consistent": False,
                "symbolic_correct": False,
                "boxed_answer": boxed_ans,
                "error": lemma_rep.get("error_message") or "Lemma inconsistency",
                "format_penalty": 0.0,
                "lemma_penalty": lemma_rep.get("penalty", -1.5),
            }

        sym_valid, sym_err = self.verify_sympy_solution(
            boxed_ans, ground_truth, equation_str=equation_str
        )

        return {
            "valid": sym_valid,
            "scaffold_adherent": scaffold_rep["adherent"],
            "lemma_consistent": True,
            "symbolic_correct": sym_valid,
            "boxed_answer": boxed_ans,
            "error": sym_err if not sym_valid else None,
            "format_penalty": 0.0,
            "lemma_penalty": 0.0,
        }

    def verify_completion(
        self,
        completion: str,
        ground_truth: str,
        equation_str: Optional[str] = None
    ) -> Dict[str, Any]:
        """Alias for verify pipeline."""
        return self.verify(completion, ground_truth, equation_str=equation_str)
