"""Detect packages upgraded by pip while older modules remain in a notebook kernel."""
from importlib import metadata
import platform
import sys

VERSIONED_MODULES = {
    "Pillow": ("PIL", "PIL.Image"),
    "numpy": ("numpy",),
    "torch": ("torch",),
    "torchvision": ("torchvision",),
    "scipy": ("scipy",),
    "scikit-image": ("skimage",),
    "transformers": ("transformers",),
}


class StaleNotebookPackagesError(RuntimeError):
    """Restart is required to use the newly installed packages consistently."""


def runtime_info():
    """Do not import ML libraries or initialize a device to collect versions."""
    packages = {}
    for distribution, module_names in VERSIONED_MODULES.items():
        try:
            installed = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            installed = None
        loaded = {}
        for name in module_names:
            module = sys.modules.get(name)
            value = getattr(module, "__version__", None)
            if value is not None:
                loaded[name] = str(value)
        if distribution == "Pillow":
            core = sys.modules.get("PIL._imaging")
            value = getattr(core, "PILLOW_VERSION", None)
            if value is not None:
                loaded["PIL._imaging"] = str(value)
        packages[distribution] = {"installed": installed, "loaded": loaded}
    return {"python": platform.python_version(), "executable": sys.executable, "packages": packages}


def assert_runtime_matches_installed():
    """Fail early after pip upgrades; reloading Pillow submodules is not a safe repair."""
    info = runtime_info()
    mismatches = []
    for distribution, versions in info["packages"].items():
        installed = versions["installed"]
        if installed is None:
            continue
        for name, loaded in versions["loaded"].items():
            # PyPI torch distributions can omit the +cu... suffix in metadata
            # while torch.__version__ includes it. That is not a stale import.
            compatible = loaded == installed or (
                "+" not in installed and loaded.split("+", 1)[0] == installed
            )
            if not compatible:
                mismatches.append(f"{name}: loaded={loaded}, installed={installed}")
    if mismatches:
        raise StaleNotebookPackagesError(
            "pip updated packages that are still loaded in this notebook kernel:\n"
            + "\n".join(mismatches)
            + "\nRestart the Kaggle kernel/session, then run the notebook from the first cell. "
              "No Python downgrade, weight download or dataset deletion is needed. "
              "Do not repair Pillow by reloading individual PIL submodules."
        )
    return info
