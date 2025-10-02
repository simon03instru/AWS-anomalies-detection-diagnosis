"""
Setup script for the CAMEL AI Event-Driven Agent System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="camel-event-driven-agents",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Event-driven multi-agent system using CAMEL AI and Kafka",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/camel-event-driven-agents",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "black>=24.1.1",
            "flake8>=7.0.0",
        ],
    },
)