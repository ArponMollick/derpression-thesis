# Depression Detection from Social Media Analytics

Machine learning project for depression detection from social media text.

## Requirements

* Python 3.12
* Git
* Virtual environment
* Required Python packages

## Setup

```bash
git clone <repository-url>
cd depression-thesis

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again.

## Dataset

Place the dataset files inside the project's dataset directory.

The current dataset uses:

```text
clean_text
is_depression
```

with:

```text
0 = Non-Depression
1 = Depression
```

## Run

Run the scripts in order:

```text
1. Data inspection
2. Data cleaning and splitting
3. TF-IDF
4. Word2Vec
5. GloVe
6. TF-IDF + VADER
7. TF-IDF + VADER + Psycholinguistic
```

Each experiment trains the configured machine-learning models.

## GloVe

Download the required pre-trained GloVe embeddings and place them in the path specified by the GloVe script.

## Output

Generated experiment outputs and CSV files are stored in the `results/` directory.
