from setuptools import setup, find_packages

setup(
    name='burst_attn',
    version='0.1.0',
    author='MayDomine',
    author_email='why1.2seed@gmail.com',
    description='Multi-gpu attention implementation: Burst-Attn',
    install_requires=[
        'pynvml',
        'einops',
        "triton==2.0.0.dev20221202",
        'flash-attn'
    ]
)