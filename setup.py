from setuptools import setup

setup(
    name="smpl-lang",
    version="0.1.0",
    py_modules=["interpreter", "smpl_lists"],  # Add any other modules here
    entry_points={
        'console_scripts': [
            'smpl=interpreter:cli',
        ]
    },
    install_requires=[],
)
