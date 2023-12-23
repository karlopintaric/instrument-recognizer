from setuptools import setup, find_packages

setup(
    name="lumen-irmas",
    version="0.1.0",
    description="LUMEN Data Science nagradni zadatak",
    author="Karlo Pintaric i Tatjana Cigula",
    packages=find_packages(include=["src"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy==1.23.5",
        "transformers==4.27.4",
    ],
    extras_require={
        "backend": ["fastapi==0.95.1", "uvicorn==0.21.1", "pydantic==1.10.7", "python-multipart==0.0.6"],
        "frontend": ["streamlit==1.21.0", "requests==2.28.2", "soundfile==0.12.1"],
        "user": [
            "lumen-irmas[backend]",
            "lumen-irmas[frontend]",
            "torch==1.13.1",
            "torchaudio==0.13.1",
            "torchvision==0.14.1",
            ],
        "dev": [
            "lumen-irmas[user]",
            "librosa==0.10.0.post2",
            "pandas==1.5.3",
            "scikit-learn==1.2.2",
            "tqdm==4.65.0",
            "wandb==0.14.2",
            "pytest==7.3.1",
            "joblib==1.2.0",
            "PyYAML==6.0",
            "flake8==6.0.0",
            "isort== 5.12.0",
            "black==23.3.0"
            ]
    },
)
