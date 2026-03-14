# Contributing to Smart City ANPR System

Thank you for your interest in contributing! This is an M.Tech research project
at SRM Institute of Science and Technology and welcomes community contributions.

---

## 🚀 Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork:
   ```bash
   git clone https://github.com/nskitechgmail/mtech_anpr_project.git
   cd mtech_anpr_project
   ```
3. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Set up the environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # Linux / Mac
   pip install -r requirements.txt
   ```
5. **Make your changes** and add tests
6. **Run tests** to confirm nothing is broken:
   ```bash
   pytest tests/test_suite.py -v -m unit
   ```
7. **Push** and open a **Pull Request**

---

## 🧪 Running Tests

```bash
# All 30 tests
pytest tests/test_suite.py -v

# Unit tests only (no GPU required)
pytest tests/test_suite.py -v -m unit

# Integration tests
pytest tests/test_suite.py -v -m integration

# With coverage report
pytest tests/test_suite.py --cov=. --cov-report=term-missing
```

Target coverage: **≥ 84%** (core modules). Please keep coverage at or above this
level when submitting a pull request.

---

## 📐 Code Style

- Follow **PEP 8** for all Python code
- Add **docstrings** to all public classes and functions
- Use **type hints** on function signatures where possible
- Keep line length ≤ 100 characters
- Run `python -m py_compile <file>` before committing to catch syntax errors

---

## 🐛 Reporting Bugs

Please open a **GitHub Issue** and include:

- Python version (`python --version`) and OS
- Steps to reproduce the bug
- Expected behaviour vs actual behaviour
- Full error traceback (copy from terminal)
- Whether you have a GPU or are running CPU-only

---

## 💡 Areas for Contribution

| Area | Description |
|---|---|
| Indian plate formats | Additional state/UT plate regex patterns |
| Safety classifier | Fine-tuned MobileNetV3 weights on Indian traffic data |
| Multi-camera sync | Cross-camera vehicle re-identification |
| Edge deployment | Jetson Nano / Raspberry Pi optimisation |
| Dashboard UI | Additional metrics, dark/light theme toggle |
| Alert channels | WhatsApp / Telegram bot integration |
| Dataset scripts | Augmentation scripts for more environmental conditions |
| Documentation | Tutorials, API usage examples, Postman collections |

---

## 📁 Project Structure Quick Reference

```
├── main.py                   # Entry point (CLI)
├── config/settings.py        # All runtime configuration
├── core/
│   ├── pipeline.py           # Capture loop + orchestration
│   └── plate_recogniser.py   # 5-stage recognition pipeline
├── models/model_manager.py   # Lazy-loaded model registry
├── ui/dashboard.py           # Tkinter GUI dashboard
├── api/server.py             # FastAPI REST API
├── utils/                    # Annotator, reporter, heatmap, alerts, anonymiser
└── tests/test_suite.py       # 30 unit + integration tests
```

---

## 📬 Contact

| Name | Email | Role |
|---|---|---|
| Sathish Kumar Nagalingam | sv2447@srmist.edu.in | Lead Developer |
| S. Venkatesh | venkates9@srmist.edu.in | Co-Researcher |

Department of Computational Intelligence · SRM IST Chennai · 2024–25
