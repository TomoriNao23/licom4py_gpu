"""
File: loader.py
Description: Generic loading mixin for configuration classes, providing
    file-based and package-based configuration loading capabilities.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-03
"""

# Standard library imports
import configparser
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Optional, Type, Union, get_args, get_origin


class LoaderMixin:
    """Generic loading/creation mixin to be inherited by config classes."""

    @classmethod
    def create(cls: Type, path: Optional[Union[str, Path]] = None):
        if path is not None:
            return cls._load_from_file(Path(path))
        return cls._load_from_package_file(anchor_file=Path(__file__))

    # Map Python types to ConfigParser getters
    _TYPE_GETTER = {
        int: configparser.ConfigParser.getint,
        str: configparser.ConfigParser.get,
        float: configparser.ConfigParser.getfloat,
        bool: configparser.ConfigParser.getboolean,
    }

    @classmethod
    def _parse_fields_from_config(
        cls: Type, config: configparser.ConfigParser
    ) -> dict[str, Any]:
        """Parse dataclass init fields for 'cls' from a ConfigParser."""
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if not f.init:
                continue
            section = f.metadata.get("section") if f.metadata is not None else None
            if section is None:
                continue

            f_type = f.type
            origin = get_origin(f_type)
            if origin is Union:
                args = tuple(a for a in get_args(f_type))
                if len(args) == 2 and type(None) in args:
                    actual_type = args[0] if args[1] is type(None) else args[1]
                    getter = cls._TYPE_GETTER[actual_type]
                    if config.has_option(section, f.name):
                        kwargs[f.name] = getter(config, section, f.name)
                    else:
                        kwargs[f.name] = f.default
                else:
                    getter = cls._TYPE_GETTER[f_type]
                    kwargs[f.name] = getter(config, section, f.name)
            else:
                getter = cls._TYPE_GETTER[f_type]
                if config.has_option(section, f.name):
                    kwargs[f.name] = getter(config, section, f.name)
                elif f.default != MISSING and f.default_factory == MISSING:
                    kwargs[f.name] = f.default
                elif f.default_factory != MISSING:
                    kwargs[f.name] = f.default_factory()
                else:
                    kwargs[f.name] = getter(config, section, f.name)

        return kwargs

    @classmethod
    def _load_from_file(cls: Type, path: Path) -> Any:
        """Instantiate dataclass 'cls' from a configuration file."""
        config = configparser.ConfigParser()
        config.read(str(path), encoding="utf-8")
        kwargs = cls._parse_fields_from_config(config)
        return cls(**kwargs)

    @classmethod
    def _load_from_package_file(
        cls: Type, filename: str = "namelist", anchor_file: Optional[Path] = None
    ) -> Any:
        """
        Instantiate dataclass 'cls' from the project-level scripts/ directory.
        Walks up 4 levels from loader.py (readnamelist/ → licom/ → src/ → project root)
        then resolves <root>/scripts/<filename>.
        """
        base = (
            anchor_file.resolve()
            if anchor_file is not None
            else Path(__file__).resolve()
        )
        project_root = base
        for _ in range(4):
            project_root = project_root.parent
        full_path = project_root / "scripts" / filename
        return cls._load_from_file(full_path)
