import json
import re
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

# LLM for extracting the math problem intent in JSON
extractor_llm = ChatOllama(
    model="gemma3:4b",
    temperature=0,
    format="json" # Forces valid JSON output
)

# Configures SymPy parser to handle '3x' or 'x^2' if the LLM slips up
transformations = standard_transformations + (
    implicit_multiplication_application,
    convert_xor
)


def safe_parse(expr_str: str, local_dict: dict | None = None):
    """Safely parses a math string into a SymPy expression."""
    if expr_str is None:
        return None
    base_locals = {"oo": sp.oo, "pi": sp.pi, "e": sp.E, "E": sp.E}
    if local_dict:
        base_locals.update(local_dict)
    return parse_expr(expr_str.strip(), local_dict=base_locals, transformations=transformations)


def extract_math_intent(question: str, prompt_path: str = "intent_extraction_prompt.txt") -> dict:
    """Uses the LLM to convert a natural language math question into JSON."""
    system_prompt = Path(prompt_path).read_text()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question:\n{question}")
    ]
    intent_response = extractor_llm.invoke(messages)
    return json.loads(str(intent_response.content))


def solve_with_sympy(intent: dict):
    """Executes the symbolic calculation in SymPy based on the extracted JSON intent."""
    op = intent.get("operation", "unsupported")
    expr_str = intent.get("expression")
    var_str = intent.get("variable") or "x"

    if op in ("unsupported", "none") or not expr_str:
        return None

    var = sp.Symbol(var_str)
    local_vars = {var_str: var}
    expr = safe_parse(expr_str, local_vars)

    if op == "derivative":
        order = int(intent.get("order") or 1)
        result = sp.diff(expr, var, order)
        if intent.get("evaluation_point") is not None:
            eval_pt = safe_parse(str(intent["evaluation_point"]), local_vars)
            result = result.subs(var, eval_pt)
        return sp.simplify(result)

    elif op == "integral":
        lower = intent.get("lower_bound")
        upper = intent.get("upper_bound")
        if lower is not None and upper is not None:
            low_val = safe_parse(str(lower), local_vars)
            up_val = safe_parse(str(upper), local_vars)
            result = sp.integrate(expr, (var, low_val, up_val))
        else:
            result = sp.integrate(expr, var)
        return sp.simplify(result)

    elif op == "limit":
        eval_pt = safe_parse(str(intent.get("evaluation_point", "0")), local_vars)
        direction = intent.get("limit_direction") or "+-"
        return sp.limit(expr, var, eval_pt, dir=direction)

    elif op == "solve":
        # Returns the set/list of solutions for expr = 0
        solutions = sp.solve(expr, var)
        return solutions

    elif op == "evaluate":
        if intent.get("evaluation_point") is not None:
            eval_pt = safe_parse(str(intent["evaluation_point"]), local_vars)
            return sp.simplify(expr.subs(var, eval_pt))
        return sp.simplify(expr)

    elif op == "simplify":
        return sp.simplify(expr)

    return None


def extract_final_answer_from_response(response_text: str) -> str | None:
    """Extracts the expression following 'FINAL_ANSWER:' from the RAG output."""
    match = re.search(r"FINAL_ANSWER:\s*(.+)", response_text, re.IGNORECASE)
    if match:
        ans = match.group(1).strip().strip("`$*.")
        # Strips constant of integration "+ C" if the LLM includes it for indefinite integrals
        ans = re.sub(r"\+\s*[cC]\b", "", ans).strip()
        return ans
    return None


def answers_are_equivalent(sympy_truth, llm_answer_str: str, var_str: str = "x") -> bool:
    """Checks if the LLM's final answer is mathematically equivalent to SymPy's result."""
    if not llm_answer_str:
        return False
    try:
        var = sp.Symbol(var_str)
        
        # Handle list of roots from sp.solve
        if isinstance(sympy_truth, list):
            return str(sympy_truth) == llm_answer_str.strip()

        candidate = safe_parse(llm_answer_str, {var_str: var})
        difference = sp.simplify(sympy_truth - candidate)
        return difference == 0
    except Exception:
        return False


if __name__ == "__main__":
    # question = "Can you solve integral from 2 to 4 of x^2 - 3x + 5 with respect to x?"
    # question = "Wtah antiderivative of x^2 - 3x + 5?"
    # question = "Find the derivative of x^3 + 2x^2 + x + 1? What is the best receipe for making chicken soup?"
    # question = "Is the limit of (sin(y)/y) as y approaches 0 1?"
    # question = "What is 1+1?"
    # question = "What is the capital of France?"
    # question = "How to remove the French language pack in Linux?"

    # NOTE: The primary variable in JSON is z here.
    question = "Let f(x, y) = (y^3)(z^2). Find the partial derivative of f with respect to z."
    print(f"Question: {question}\n")

    # Extracts JSON intent using Gemma3:4b
    intent = extract_math_intent(question)
    print("Extracted problem intent in JSON:")
    print(json.dumps(intent, indent=4))

    # Solves the problem with SymPy
    sympy_solution = solve_with_sympy(intent)
    print(f"\nSymPy verified solution: {sympy_solution}")
