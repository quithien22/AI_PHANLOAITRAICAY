from setuptools import find_packages, setup

setup(
    name="fruit-classifier",
    version="1.1.0",
    description="Phân loại trạng thái trái cây (fresh/rotten) – huấn luyện và REST API",
    packages=find_packages(include=["src", "src.*", "api", "api.*"]),
    python_requires=">=3.9",
    install_requires=[
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "pillow",
        "numpy",
        "tensorflow>=2.12",
    ],
    extras_require={
        # pip install -e ".[train]"
        "train": ["pandas", "matplotlib", "seaborn", "scikit-learn", "jupyter"],
        "dev": ["pytest"],
    },
)
