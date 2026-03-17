# Create a Python virtual environment
python -m venv .venv

# Activate it
& .venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

Write-Host "Environment setup complete."
