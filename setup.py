from setuptools import setup

setup(
    name="lumen-irmas",
    version="0.1.0",
    install_requires=[
        "librosa==0.10.0.post2",
        "numpy==1.23.5",
        "pandas==2.0.0",
        "scikit-learn==1.2.2",
        "torch==2.0.0",
        "torchaudio==2.0.1",
        "torchvision==0.15.1",
        "tqdm==4.65.0",
        "transformers==4.27.4",
        "wandb==0.14.2"
        ],
    packages=["lumen_irmas"],
    extras_require={
        "all": ["matplotlib", "fastapi[all]", "streamlit"],
    },
)