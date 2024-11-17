from pathlib import Path

__version__ = '0.1.0'
__author__ = 'Timo Ehrle'

# This init file automatically builds a module list.  This allows one to put new .py files in this workflow directory
# and make them automatically accessible in other code that uses this package: "from pytorch_utils import *"

# find all .py files in current directory
pyFiles = Path(__file__).parent.glob("*.py")

# get the module name (i.e. no .py suffix) and save in __all__ so that "import *" can be used
__all__ = [Path(f).stem for f in pyFiles if Path(f).name != '__init__.py']