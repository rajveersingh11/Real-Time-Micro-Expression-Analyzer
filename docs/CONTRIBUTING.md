# Contributing to AI-MicroExpression-Analyzer

Thank you for contributing to this project! This document outlines coding standards, structural conventions, and the test suite configuration.

---

## 💻 Development Workflow

1. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Make your edits in a clean branch.
3. Validate code standards (pep8/flake8 compliance is encouraged).
4. Run the test suite before submitting pull requests.

---

## 🧪 Testing Guidelines

We utilize `pytest` for validation and testing.

### Running Tests
To run all tests locally:
```bash
pytest
```

To run a specific test script:
```bash
pytest tests/test_stress_model.py
```

### Writing New Tests
- All test scripts must reside inside the `tests/` directory.
- Use the shared fixtures defined in [conftest.py](file:///D:/Projects/portfolio/Real-Time-Micro-Expression-Analyzer/tests/conftest.py) for mock frames, mock landmarks, and pre-instantiated models.
- If testing a new model, set a fixed random seed to ensure reproducibility.

---

## 📏 Code Style and Standards
- **Logging**: Do not use raw print statements in production code. Use the centralized logger:
  ```python
  from src.utils.logger import get_logger
  logger = get_logger("my_module")
  logger.info("Message")
  ```
- **Error Handling**: Use custom exceptions defined in [exceptions.py](file:///D:/Projects/portfolio/Real-Time-Micro-Expression-Analyzer/src/utils/exceptions.py).
- **Imports**: Avoid relative/bare imports. Always use absolute imports matching the package name (`src.*`).
- **Typing**: Add type hints for new functions and public API classes where appropriate.
