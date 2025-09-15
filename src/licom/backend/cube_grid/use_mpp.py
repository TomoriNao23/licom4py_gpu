import ctypes
import os


class FMS_chtholly:
    @classmethod
    def init(cls):

        cls.load_chtholly_library()
        cls.lib.chtholly_init()

    @classmethod
    def load_chtholly_library(cls):
        """Load the Chtholly FMS library"""
        # Get the library path
        # Get the project root directory (go up 5 levels from this file)
        # Current file: src/licom/backend/cube_grid/use_mpp.py
        # Go up 5 levels: cube_grid -> backend -> licom -> src -> project_root
        lib_path = os.path.join("/Users/Chtholly/Documents/2025/code/modes/chtholly_try/lib/lib/libfms_unified.dylib")
      
        # Load the library
        cls.lib = ctypes.CDLL(lib_path)
        
        # Define function signatures
        cls.lib.chtholly_init.argtypes = []
        cls.lib.chtholly_init.restype = None
        
        cls.lib.chtholly_end.argtypes = []
        cls.lib.chtholly_end.restype = None

    @classmethod
    def end(cls):
        """End the Chtholly FMS library"""
        if hasattr(cls, 'lib'):
            cls.lib.chtholly_end()