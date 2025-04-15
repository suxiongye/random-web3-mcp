from setuptools import setup, find_packages

setup(
    name="randomweb3mcp",
    version="0.1.0",
    package_dir={"": "."},
    packages=find_packages(where="."),
    install_requires=[
        "httpx>=0.28.1",
        "numpy>=2.2.4",
        "mcp>=1.6.0",
        "web3>=7.10.0",
    ],
    python_requires=">=3.12",
) 