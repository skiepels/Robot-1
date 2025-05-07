from setuptools import setup, find_packages

setup(
    name="trading_strategy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "yfinance",
        "ib_insync",
        "python-dotenv",
        "requests",
    ],
    python_requires=">=3.7",
)