# Metadata lives in pyproject.toml. This shim only maps `tools/` and `exps/default/`
# into the package, which pyproject.toml cannot express together with package discovery.
import setuptools

setuptools.setup(
    package_dir={"yolox.tools": "tools", "yolox.exp.default": "exps/default"},
    packages=setuptools.find_packages(exclude=("tests", "tools")) + ["yolox.tools", "yolox.exp.default"],
)
