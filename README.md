📁 Project: Performance Limits of AI Surrogates for Composite Topology Optimization under Small-Data Constraints
GitHub Repo: CAD---Simulation
Author: Viswa Musunuri (M.Tech Thesis Research)

📘 Overview
This repository supports the research project titled:
"Performance Limits of AI Surrogates for Composite Topology Optimization under Small-Data Constraints"

It contains:
CAD geometry generation scripts
Simulation logs from SimScale
AI surrogate model training using Python (ANN, Random Forest)
Residual analysis and comparison metrics
Dataset and result visualizations
Paper and thesis-ready outputs

# 🏗️ CAD---Simulation: AI Surrogates for Topology Optimization of Composites

This repository supports my **M.Tech thesis and Scopus paper** titled:  
**"Residual-Based Evaluation of AI Surrogates for Composite Topology Optimization under Simulation-Only Small-Data Conditions"**

> 🔗 **Repo:** https://github.com/Viswamusunuri9/CAD---Simulation  
> 🧑‍💻 Author: Viswa Musunuri  
> 🎯 Focus: AI surrogate modeling (ANN, RF) + Simulation + Residual error analysis

---

## 📁 Project Folder Structure

CAD---Simulation/
│
├── AI_Model_Project/ # All AI models, plots, residuals, code
│ ├── *.py # ANN, RF, comparison scripts
│ ├── *.pkl # Saved trained models
│ ├── *.png # Plots (residuals, predictions)
│ ├── residuals_summary.csv # Residual analysis output
│ └── model_comparison_results.csv

├── Design Data/ # Simulation design references
│ ├── design_matrix.csv
│ └── FreeCAD + STEP files

├── dataset/ # AI-ready data
│ ├── ai_surrogate_dataset.csv
│ ├── simulation_log.csv
│ └── train-test split CSVs

├── simscale-results.zip # Raw FEA results (exported from SimScale)
├── Log Results.xlsx # Summary of simulation results
├── .gitignore # Exclude logs, pkl, etc.
└── README.md # You are here


---

## ⚙️ Python Environment Setup

**Python version:** `3.11.x`  
Recommended IDE: [Visual Studio Code](https://code.visualstudio.com/)

### 🔧 Virtual Environment Setup

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

🚀 How to Run
All scripts are inside the /AI_Model_Project/ folder

1️⃣ Train and Save Models
python baseline_ann_model.py         # Basic ANN
python tuned_ann_model.py            # ANN with GridSearchCV
python rf_model.py                   # Random Forest

2️⃣ Compare All Models
python model_comparison.py           # R2, MAE, MSE + plots

3️⃣ Analyze Residual Errors
python residual_analysis.py          # Residual histograms, scatterplots, boxplots

📊 Outputs and Deliverables
Output Type	Files
🔢 Surrogate Models	*.pkl models (ANN, RF)
📈 Performance Plots	actual_vs_predicted.png, comparison_plots.png
📉 Residual Plots	residual_hist_*, residual_scatter_*, residual_boxplot_*
📄 Paper Outline	paper_outline.docx
📚 Literature Review	Literature_Matrix_2.0_Final.xlsx
📑 Tabular Logs	model_comparison_results.csv, residuals_summary.csv

🧠 Notes for Reviewers / Collaborators
All FEA simulations were performed using SimScale
CAD models were built in FreeCAD

🤝 Acknowledgements
FreeCAD: For Parametric CAD modeling
SimScale: For free FEA simulations
scikit-learn: For surrogate modeling

📫 For questions, collaboration, or journal inquiries:
Reach me via GitHub

---

## ✅ Final Suggestion
If needed, I’ll generate a `requirements.txt` and `.zip` version of this `README.md` file for quick upload.
Would you like that?
Or just copy this content directly into your repo’s README and commit.
