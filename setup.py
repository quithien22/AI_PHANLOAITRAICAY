from setuptools import find_packages, setup

setup(
    name="fruit-classifier",
    version="2.0.0",
    description="Phân loại trạng thái trái cây (fresh/rotten) bằng học máy – huấn luyện và REST API",
    packages=find_packages(include=["src", "src.*", "api", "api.*"]),
    python_requires=">=3.9",
    install_requires=[
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "pillow",
        "numpy",
        "scikit-learn",
        "joblib",
    ],
    extras_require={
        # pip install -e ".[train]"
        "train": ["pandas", "matplotlib", "seaborn", "jupyter"],
        "dev": ["pytest"],
    },
)
