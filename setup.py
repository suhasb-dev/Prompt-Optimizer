from setuptools import setup, find_packages
import pathlib

# Read the contents of README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name='gepa-optimizer',
    version='0.1.0',
    description='Universal prompt optimization framework based on GEPA',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Suhas',
    author_email='s8hasgrylls@gmail.com',
    url='https://github.com/suhas/gepa-optimizer',
    project_urls={
        'Bug Reports': 'https://github.com/suhas/gepa-optimizer/issues',
        'Source': 'https://github.com/suhas/gepa-optimizer',
        'Documentation': 'https://github.com/suhas/gepa-optimizer/docs',
    },
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'gepa>=0.0.12',
        'pandas>=1.5.0',
        'pydantic>=2.0.0',
        'python-dotenv>=1.0.0',
        'requests>=2.31.0',
        'aiohttp>=3.8.0',
        'asyncio-throttle>=1.0.0',
        'google-generativeai>=0.3.0',
        'Pillow>=9.0.0'
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.21.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
        'docs': [
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.2.0',
        ],
        'all': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.21.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.2.0',
        ]
    },
    python_requires='>=3.8',
    license='MIT',
    keywords=['prompt-optimization', 'llm', 'gepa', 'ai', 'machine-learning', 'ui-tree-extraction'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    entry_points={
        'console_scripts': [
            'gepa-optimize=gepa_optimizer.cli:main',
        ],
    },
)