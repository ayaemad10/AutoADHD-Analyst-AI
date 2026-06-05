# AutoADHD-Analyst-AI

AI-Based ADHD Analysis & Early Detection Support System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-green)

---

# AutoADHD Analyst AI

Automated AI-Powered ADHD Analysis & Early Detection Support System

AutoADHD Analyst AI is a collaborative, beginner-friendly, professional AI healthcare project designed to automate the core workflow of ADHD data analysis and early detection support. The system covers loading raw datasets, understanding data quality, profiling, exploratory analysis, preprocessing, feature engineering, machine learning modeling, evaluation, insight extraction, and dashboard reporting.

The project also teaches professional GitHub collaboration, modular architecture, pull request workflows, and team-based AI system development.

---

# Problem Statement

Many students and healthcare AI practitioners understand individual tools such as Pandas, Scikit-learn, XGBoost, and Streamlit, but struggle to combine them into one clean, reusable, collaborative AI healthcare pipeline.

Most ADHD-related projects focus only on model training without building a complete end-to-end workflow that includes preprocessing, exploratory analysis, evaluation, reporting, and deployment.

AutoADHD Analyst AI provides a structured collaborative system where teams can learn professional AI development while building a useful ADHD analysis and early detection support platform.

---

# Objectives

* Build a modular AI pipeline for ADHD analysis.
* Provide reusable starter code for each stage of the ML workflow.
* Train team members on GitHub branches, commits, pull requests, and reviews.
* Create reusable documentation for workflow planning and collaboration.
* Build a Streamlit dashboard for ADHD dataset upload and prediction.
* Generate automated visual insights and evaluation reports.

---

# Key Features

| Area                | Starter Capability                         |
| ------------------- | ------------------------------------------ |
| Data Loading        | CSV / EEG / Clinical dataset loading       |
| Data Profiling      | Missing values, duplicates, data quality   |
| EDA                 | Statistical summaries and correlations     |
| Cleaning            | Missing-value handling and outlier removal |
| Feature Engineering | Behavioral and EEG feature extraction      |
| Modeling            | Random Forest, XGBoost, SVM                |
| Evaluation          | Accuracy, Precision, Recall, F1-score      |
| Insights            | Feature importance analysis                |
| Reporting           | Markdown report generation                 |
| Dashboard           | Streamlit upload and ADHD prediction       |

---

# End-to-End System Workflow

AutoADHD Analyst AI is designed as one integrated AI pipeline. The dashboard and future AI agents should call the central pipeline instead of duplicating business logic.
flowchart TD
    A[User Dataset Upload] --> B[Data Loading]
    B --> C[Data Understanding]
    C --> D[Data Profiling]
    D --> E[EDA]
    E --> F[Data Cleaning]
    F --> G[Preprocessing]
    G --> H[Feature Engineering]
    H --> I[Model Training]
    I --> J[Model Evaluation]
    J --> K[Insight Generation]
    K --> L[Report Generation]
    L --> M[Dashboard Output]

Central pipeline file:

```python
src/autoadhd/pipeline.py
```

See:

```bash
docs/end_to_end_integration_strategy.md
```

for input/output contracts and integration rules.

---

# Folder Structure

```bash
AutoADHD-Analyst-AI/
├── app/                    # Streamlit application
├── data/                   # Raw, processed, and sample datasets
├── docs/                   # Planning and collaboration documentation
├── notebooks/              # Exploration and experiment notebooks
├── reports/                # Figures and generated reports
├── models/                 # Saved ML models
├── src/autoadhd/           # Main Python package
├── tests/                  # Automated tests
├── .github/                # GitHub templates and workflows
├── README.md
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

# Tech Stack

* Python 3.10+
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Plotly
* Streamlit
* Jupyter Notebook
* Pytest

---

# Installation

```bash
git clone https://github.com/<your-username>/AutoADHD-Analyst-AI.git

cd AutoADHD-Analyst-AI

python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

pip install -e .
```

Git Bash / macOS / Linux:

```bash
source .venv/bin/activate

pip install -r requirements.txt

pip install -e .
```

---

# Usage

Run Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```

Run tests:

```bash
pytest
```

Use pipeline in Python:

```python
from autoadhd.pipeline import PipelineConfig, run_analysis_pipeline

result = run_analysis_pipeline(
    "data/sample/example.csv",
    PipelineConfig(
        target_column="ADHD",
        model_task="classification"
    ),
)

print(result.profile)
print(result.insights)
```

---

# Team Collaboration & 8-Week Execution Plan

AutoADHD Analyst AI is organized into collaborative sub-teams. Each team has one main responsibility and one dedicated feature branch.

Every sub-team should work on its assigned branch and open Pull Requests into `develop`.

The `main` branch is only for stable releases.

Direct pushes to `main` are NOT allowed.

Weekly progress should be recorded in:

```bash
docs/weekly_updates/
```

Detailed task instructions are available in:

```bash
docs/weekly_tasks/
```

---

# Sub-Team Structure

| Sub-Team                                    | Members                                            | Branch                         |
| ------------------------------------------- | -------------------------------------------------- | ------------------------------ |
| Team 1: Project Management & GitHub         | Aya Emad                                           | feature/project-management     |
| Team 2: Data Understanding & Profiling      | Aya Ashraf + Rawya                                 | feature/data-understanding     |
| Team 3: Preprocessing & Feature Engineering | Aya Emad + Amal Wagih                              | feature/preprocessing-features |
| Team 4: Machine Learning Modeling           | Aya Ashraf + Amal Sherif + Aya Emad                | feature/modeling               |
| Team 5: Evaluation & Insights               | Amal Wagih                                         | feature/evaluation-insights    |
| Team 6: Reporting & Dashboard               | Rawya + Aya Emad + Amal + Amal Sherif + Aya Ashraf | feature/reporting-dashboard    |

---

# Team Collaboration Workflow

AutoADHD Analyst AI uses a beginner-friendly professional GitHub workflow.

---

# Branch Policy

| Branch      | Policy                               |
| ----------- | ------------------------------------ |
| main        | Stable production version only       |
| develop     | Integration branch for reviewed work |
| feature/... | Team feature branches                |

---

# Pull Request Policy

1. Start from the latest `develop` branch.
2. Create a feature branch.
3. Make focused commits.
4. Push changes to GitHub.
5. Open Pull Request into `develop`.
6. Request review before merge.
7. Resolve conflicts before merging.

---

# Branch Strategy

| Branch Type | Example                   | Purpose                  |
| ----------- | ------------------------- | ------------------------ |
| Stable      | main                      | Production-ready version |
| Integration | develop                   | Combined reviewed work   |
| Feature     | feature/modeling          | New functionality        |
| Fix         | fix/preprocessing-bug     | Bug fixes                |
| Docs        | docs/update-roadmap       | Documentation            |
| Experiment  | experiment/xgboost-tuning | Temporary experiments    |

---

# Roadmap

### Week 1

Project setup and GitHub onboarding

### Week 2

Dataset understanding and profiling

### Week 3

EDA and visualization

### Week 4

Preprocessing and feature engineering

### Week 5

Machine learning modeling

### Week 6

Evaluation and insights

### Week 7

Dashboard and reporting

### Week 8

Final integration and presentation

---

# Future Improvements

* EEG signal processing
* Explainable AI (SHAP)
* PDF medical report generation
* Deep learning models
* ADHD risk prediction
* Real-time dashboard
* Docker deployment
* CI/CD integration

---

# Contributors

| Name        | Role                        |
| ----------- | --------------------------- |
| Aya Emad    | Project Management & GitHub |
| Aya Ashraf  | Data Analysis & Modeling    |
| Rawya       | Reporting & Dashboard       |
| Amal Wagih  | Preprocessing & Evaluation  |
| Amal Sherif | Machine Learning            |

---

# License

This project is licensed under the MIT License.

See `LICENSE` for more information.
