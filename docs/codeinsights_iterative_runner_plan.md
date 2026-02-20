# Plan: Iterative LLM Code Generation System for CodeInsights

## Goal
Build a system that simulates student-like iterative coding behavior with LLMs, generating a dataset of interaction histories (attempts, test results, iterations) matching the student dataset format.

## Approach: Extend HELM
HELM's standard pipeline is single-shot (no iteration support). We will extend HELM by creating a new `IterativeRunner` that:
- Uses HELM's `LocalContext` for LLM calls (benefits from caching, credential management)
- Uses HELM's `CPPEvaluator` for test execution
- Adds a custom iteration loop around these components

## Starting Point: "LLM tries to get full score" (CorrectCode scenario with iteration)

---

## Key Design Decisions

| Decision | Choice |
|----------|--------|
| **Test visibility during iteration** | PUBLIC only (first N tests) - mimics real student experience |
| **Public/Private separation** | First N tests are public (configurable, e.g., first 3 of 10) |
| **Stopping criteria** | Stop when all PUBLIC tests pass OR max iterations reached |
| **Final evaluation** | Run ALL tests (public + private) for final score |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    IterativeCodeRunner                          │
├─────────────────────────────────────────────────────────────────┤
│  For each question:                                             │
│    1. Generate initial code (prompt LLM)                        │
│    2. Run PUBLIC tests (CPPEvaluator)                           │
│    3. If score < 1.0 and iterations < N:                        │
│       - Build feedback prompt with test results                 │
│       - Generate new code                                       │
│       - Repeat from step 2                                      │
│    4. Run ALL tests (final evaluation)                          │
│    5. Log iteration history                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Files to Create/Modify

### 1. New File: `src/helm/benchmark/codeinsights_iterative_runner.py`
Main runner that handles the iteration loop.

```python
class IterativeCodeRunner:
    def __init__(
        self,
        model_deployment: str,      # e.g., "openai/gpt-4"
        max_iterations: int = 5,
        num_public_tests: int = 3,  # First N tests shown during iteration
        timeout_seconds: int = 10,
        output_path: str = "iterative_results",
        base_path: str = "prod_env",  # For credentials
    )

    def run(self, questions: List[Question]) -> List[SessionLog]
    def _make_request(self, prompt: str) -> str
    def _run_public_tests(self, code: str, ...) -> TestResult  # Only first N
    def _run_all_tests(self, code: str, ...) -> TestResult     # All tests (final eval)
    def _build_feedback_prompt(self, code: str, test_results: TestResult) -> str

    def _iterate(self, question: Question) -> SessionLog:
        """Main iteration loop for one question"""
        public_tests = question.test_cases[:self.num_public_tests]
        all_tests = question.test_cases

        for attempt in range(self.max_iterations):
            code = self._make_request(prompt)
            results = self._run_public_tests(code, public_tests, ...)

            self._log_attempt(attempt, "precheck", code, results)

            if results.score == 1.0:  # All PUBLIC tests pass
                break

            prompt = self._build_feedback_prompt(code, results)

        # Final evaluation on ALL tests
        final_results = self._run_all_tests(code, all_tests, ...)
        self._log_attempt(attempt, "check", code, final_results)

        return session_log
```

### 2. New File: `src/helm/benchmark/codeinsights_data_loader.py`
Load questions from HuggingFace dataset.

```python
class CodeInsightsDataLoader:
    def __init__(self, num_public_tests: int = 3):
        self.num_public_tests = num_public_tests  # First N tests are public

    def load_questions(self, scenario: str = "correct_code") -> List[Question]
    def load_test_cases(self) -> Dict[str, List[TestCase]]

    def split_tests(self, test_cases: List[TestCase]) -> Tuple[List, List]:
        """Split into public (first N) and private (rest)"""
        return test_cases[:self.num_public_tests], test_cases[self.num_public_tests:]
```

### 3. Modify: `src/helm/benchmark/metrics/codeinsights_correct_code_metrics.py`
Enhance `CPPEvaluator` to return detailed output (not just pass/fail).

```python
class CPPEvaluator:
    def evaluate_with_details(self, code: str) -> DetailedTestResult:
        # Returns actual output for failed tests (for feedback)
```

### 4. New File: `src/helm/benchmark/codeinsights_session_log.py`
Data structures for logging iteration history.

```python
@dataclass
class AttemptLog:
    attempt_id: int
    timestamp: str
    response_type: str  # "precheck" or "check"
    code: str
    test_results: List[bool]
    pass_rate: float

@dataclass
class SessionLog:
    question_id: str
    model: str
    attempts: List[AttemptLog]
    final_score: float
    total_iterations: int
```

---

## Implementation Steps

### Step 1: Enhance CPPEvaluator for detailed feedback
- Modify `evaluate()` to capture actual output (not just pass/fail)
- Return structured result with expected vs actual for failed tests
- This enables meaningful feedback to the LLM

### Step 2: Create data loader
- Load questions from `Scenario1_2_data.csv`
- Parse test cases from `question_unittests` field
- Separate public vs private tests (using precheck info if available)

### Step 3: Create iterative runner
- Initialize `LocalContext` for LLM calls
- Loop: generate → test (first N) → feedback → retry
- Use only PUBLIC tests (first N) for iteration feedback
- Stop when: all public tests pass OR max iterations reached
- Run ALL tests for final evaluation (to get true score)

### Step 4: Implement feedback prompt builder
```python
def _build_feedback_prompt(self, original_prompt, code, test_results):
    return f"""
{original_prompt}

Your previous attempt:
```cpp
{code}
```

Test Results:
{self._format_test_results(test_results)}

Please fix the issues and provide corrected code.
"""
```

### Step 5: Create session logger
- Output format matching student dataset structure
- Fields: question_id, attempt_id, timestamp, response_type, code, pass pattern
- Export to CSV/JSON

---

## Output Dataset Format

Match the student `main_data.csv` format:

| Column | Description |
|--------|-------------|
| `model_id` | LLM identifier (replaces student_id) |
| `question_unittest_id` | Question ID |
| `attempt_id` | Iteration number |
| `timestamp` | When attempt was made |
| `response_type` | "precheck" (public tests) or "check" (all tests) |
| `response` | Generated code |
| `pass` | Pass pattern (e.g., "11101") |

---

## Entry Point Script

```bash
# Run iterative generation
python -m helm.benchmark.run_codeinsights_iterative \
    --model openai/gpt-4 \
    --max-iterations 5 \
    --scenario correct_code \
    --output results/gpt4_iterations.csv
```

---

## Future Extensions (4 scenarios)

1. **CorrectCode** (this plan): LLM iterates to get full score
2. **StudentCoding**: LLM iterates while mimicking student style
3. **StudentMistake**: LLM iterates but intentionally makes student-like errors
4. **CodeEfficiency**: LLM iterates while matching student's efficiency level

Each scenario would use the same iteration infrastructure but with different:
- Initial prompts
- Feedback formatting
- Stopping criteria

---

## Verification

1. Run on small subset (5 questions) to verify iteration loop works
2. Check that test execution matches expected pass/fail
3. Verify output CSV format matches student dataset
4. Compare iteration patterns to real student data
