import os
import yaml
import copy
import json
from typing import Any
from pathlib import Path
from omegaconf import OmegaConf
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from ._errors import ConfigurationError
from ._contracts import ComponentType, REQUIRED_COMPONENT_TYPES, Task, _get_ctype

_TARGETS = frozenset({"train", "val"})
_COMPONENT_FIELDS = frozenset({"ctype", "type", "params", "components"})
_VARIANT_COMPONENT_TYPES = frozenset({ComponentType.DATALOADER})
_CONFIG_SUFFIX = frozenset({".yml", ".yaml"})
CONFIGNAME = "config.yml"


def _validate_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a mapping")
    return value


def _validate_nested_components(
    value: object,
    path: str,
) -> Mapping[str, Any] | list[Any]:
    if isinstance(value, Mapping) or isinstance(value, list):
        return value
    raise ConfigurationError(f"{path} must be a mapping or list")


def _parse_nested_components(
    data: Mapping[str, Any] | list[Any],
    config_path: str,
    parent_ctype: ComponentType,
) -> dict[str, "ComponentConfig"] | list["ComponentConfig"]:

    if isinstance(data, list):
        parsed_items: list[ComponentConfig] = []
        for index, config in enumerate(data):
            item_path = f"{config_path}[{index}]"
            config = _validate_mapping(config, item_path)
            ctype = _get_ctype(config.get("ctype", parent_ctype), item_path)
            parsed_items.append(
                ComponentConfig.from_dict(config, config_path=item_path, ctype=ctype)
            )
        return parsed_items

    parsed_components: dict[str, ComponentConfig] = {}
    for name, config in data.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError(f"{config_path} keys must be non-empty strings")

        config = _validate_mapping(config, f"{config_path}.{name}")

        try:
            ctype = ComponentType(name)
        except ValueError:
            ctype = _get_ctype(config.get("ctype", parent_ctype))

        parsed_components[name] = ComponentConfig.from_dict(
            config, config_path=f"{config_path}.{name}", ctype=ctype
        )
    return parsed_components


def _iter_components(
    components: Mapping[str, "ComponentConfig"] | list["ComponentConfig"],
) -> Iterable["ComponentConfig"]:
    if isinstance(components, Mapping):
        return components.values()
    return components


def _components_to_dict(
    components: Mapping[str, "ComponentConfig"] | list["ComponentConfig"],
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(components, Mapping):
        return {role: config.to_dict() for role, config in components.items()}
    return [config.to_dict() for config in components]


@dataclass(slots=True)
class ComponentVariantConfig:
    params: dict[str, Any] = field(default_factory=dict)
    components: dict[str, "ComponentConfig"] | list["ComponentConfig"] = field(
        default_factory=dict
    )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        config_path: str,
        ctype: ComponentType,
    ) -> "ComponentVariantConfig":
        data = _validate_mapping(data, config_path)

        unknown = set(data).difference({"params", "components"})
        if unknown:
            keys = ", ".join(sorted(unknown))
            raise ConfigurationError(f"{config_path} contains unknown fields: {keys}")

        params = data.get("params", {})
        params = _validate_mapping(params, f"{config_path}.params")

        children = data.get("components", {})
        children = _validate_nested_components(children, f"{config_path}.components")

        return cls(
            params=dict(params),
            components=_parse_nested_components(
                children, config_path=f"{config_path}.components", parent_ctype=ctype
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.params:
            result["params"] = dict(self.params)
        if self.components:
            result["components"] = _components_to_dict(self.components)
        return result


@dataclass(frozen=True, slots=True)
class ComponentConfig:
    """Config for a public component and nested settings owned by its builder."""

    ctype: ComponentType
    type: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    components: dict[str, "ComponentConfig"] | list["ComponentConfig"] = field(
        default_factory=dict
    )
    variants: dict[str, ComponentVariantConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        ctype: ComponentType | str,
        config_path: str = "component",
    ) -> "ComponentConfig":
        data = _validate_mapping(data, config_path)

        if ctype is None:
            raise ConfigurationError(f"{path} must be parsed from a component type key")
        ctype = _get_ctype(ctype, f"{config_path} key")

        unknown_keys = [key for key in data if key not in _COMPONENT_FIELDS]
        if unknown_keys:
            if ctype not in _VARIANT_COMPONENT_TYPES:
                keys = ", ".join(key for key in unknown_keys)
                raise ConfigurationError(
                    f"{config_path} contains unknown fields: {keys}"
                )

            if set(unknown_keys).difference(_TARGETS):
                keys = ", ".join(key for key in unknown_keys)
                raise ConfigurationError(f"{config_path} contains unknown keys: {keys}")

        registry_name = data.get("type")
        if not isinstance(registry_name, str) or not registry_name.strip():
            raise ConfigurationError(f"{config_path}.type must be a non-empty string")

        params = data.get("params", {})
        params = _validate_mapping(params, f"{config_path}.params")

        children = data.get("components", {})
        children = _validate_nested_components(children, f"{config_path}.components")

        # Отдельная логика обработки dataloader, т.к. разные параметры на train | val
        variants: dict[str, ComponentVariantConfig] = {}
        for target in unknown_keys:
            variants[target] = ComponentVariantConfig.from_dict(
                data[target],
                config_path=f"{config_path}.{target}",
                ctype=ctype,
            )

        return cls(
            ctype=ctype,
            type=registry_name,
            params=dict(params),
            components=_parse_nested_components(
                children, config_path=f"{config_path}.components", parent_ctype=ctype
            ),
            variants=variants,
        )

    def variant(self, name: str) -> "ComponentConfig":
        try:
            variant = self.variants[name]
        except KeyError as err:
            raise ConfigurationError(
                f"{self.ctype.value} does not define variant {name!r}"
            ) from err

        duplicate_params = set(self.params).intersection(variant.params)
        if duplicate_params:
            keys = ", ".join(sorted(duplicate_params))
            raise ConfigurationError(
                f"{self.ctype.value}.{name} params override common params: {keys}"
            )

        if not isinstance(self.components, Mapping) or not isinstance(
            variant.components, Mapping
        ):
            raise ConfigurationError(
                f"{self.ctype.value}.{name} components must be mappings for variants"
            )

        duplicate_components = set(self.components).intersection(variant.components)
        if duplicate_components:
            keys = ", ".join(sorted(duplicate_components))
            raise ConfigurationError(
                f"{self.ctype.value}.{name} components override common components: {keys}"
            )

        params = dict(self.params)
        params.update(variant.params)

        components = dict(self.components)
        components.update(variant.components)

        return ComponentConfig(
            ctype=self.ctype,
            type=self.type,
            params=params,
            components=components,
        )

    def component_types(self) -> set[ComponentType]:
        component_types = {self.ctype}
        for component in _iter_components(self.components):
            component_types.update(component.component_types())
        for variant in self.variants.values():
            for component in _iter_components(variant.components):
                component_types.update(component.component_types())
        return component_types

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.type is not None:
            result["type"] = self.type
        if self.params:
            result["params"] = dict(self.params)
        if self.components:
            result["components"] = _components_to_dict(self.components)
        for role, variant in self.variants.items():
            result[role] = variant.to_dict()
        return result


@dataclass
class ExperimentConfig:
    """Declarative component graph plus task-independent runtime options."""

    name: str
    task: Task
    outdir: str | Path | None
    components: dict[ComponentType, ComponentConfig]
    runtime: dict[str, Any] = field(default_factory=dict)
    saving: list[dict] | None = None

    @property
    def project(self) -> str:
        return self.name

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        data = _validate_mapping(data, "experiment config")

        name: str = data.get("name", None)
        if not isinstance(name, str) and name.strip():
            raise ConfigurationError("`name` must be a non-empty string when specified")

        try:
            task = Task(data.get("task"))
        except ValueError as err:
            raise ConfigurationError(str(err)) from err

        # Could be passed from argparse
        outdir = data.get("outdir", None)
        if outdir and not os.path.isdir(outdir):
            raise ConfigurationError("Provided `outdir` must be actual dir")

        # Has default policy
        saving_policy = data.get("saving", None)
        if saving_policy:
            _validate_nested_components(saving_policy, "saving")

        # Must be set
        components = data.get("components")
        _validate_mapping(components, "components")
        if not components:
            raise ConfigurationError(
                "experiment components must be a non-empty mapping"
            )

        parsed_components: dict[ComponentType, ComponentConfig] = {}
        for raw_ctype, component in components.items():
            ctype = _get_ctype(raw_ctype, f"components.{raw_ctype}")
            parsed_components[ctype] = ComponentConfig.from_dict(
                component,
                ctype=ctype,
                config_path=f"components.{ctype.value}",
            )

        runtime = data.get("runtime")
        _validate_mapping(runtime, "runtime")
        if not runtime:
            raise ConfigurationError("experiment `runtime` must be a non-empty mapping")

        cls._validate_required_components(parsed_components)
        return cls(
            name=name,
            task=task,
            outdir=outdir,
            components=parsed_components,
            runtime=runtime,
            saving=saving_policy,
        )

    @staticmethod
    def _validate_required_components(
        components: Mapping[ComponentType, ComponentConfig],
    ) -> None:
        configured_types = set(components)
        for component in components.values():
            configured_types.update(component.component_types())

        missing = REQUIRED_COMPONENT_TYPES.difference(configured_types)
        if missing:
            names = ", ".join(sorted(ctype.value for ctype in missing))
            raise ConfigurationError(
                f"experiment components are missing required types: {names}"
            )

    def require_component(self, ctype: ComponentType | str) -> ComponentConfig:
        ctype = _get_ctype(ctype, "component type")
        try:
            return self.components[ctype]
        except KeyError as err:
            raise ConfigurationError(
                f"Experiment does not define component type {ctype.value!r}"
            ) from err

    def get_component(self, ctype: ComponentType | str) -> ComponentConfig | None:
        ctype = _get_ctype(ctype, "component type")
        return self.components.get(ctype, None)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "task": self.task.value,
            "components": {
                ctype.value: component.to_dict()
                for ctype, component in self.components.items()
            },
        }

        result["name"] = self.name
        result["task"] = self.task.value
        result["saving"] = self.saving
        result["outdir"] = str(self.outdir)
        if self.runtime:
            result["runtime"] = dict(self.runtime)
        return result

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4)


def _load_yml(filepath: str | Path) -> Mapping:
    path = Path(filepath)
    if not path.exists():
        raise ConfigurationError(f"Config file does not exist: {path}")

    if path.suffix.lower() not in _CONFIG_SUFFIX:
        raise ConfigurationError(
            f"Unsupported config format {path.suffix!r}; use .yaml or .yml"
        )

    with open(filepath, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise RuntimeError("Error parsing YAML file!") from exc

    return data


def resolve_config(data: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    defaults = data.pop("defaults", [])
    if not defaults:
        return data

    merged: dict[str, Any] = {}
    for item in defaults:
        sub_conf_path = item if os.path.isfile(item) else base_dir / item
        sub_conf_data = _load_yml(sub_conf_path)
        merged = deep_merge(merged, sub_conf_data)

    final_conf = OmegaConf.create(deep_merge(merged, data))
    return OmegaConf.to_container(final_conf, resolve=True)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def load_experiment_config(filepath: str | Path) -> ExperimentConfig:
    path = Path(filepath)
    raw_data = _load_yml(path)
    resolved = resolve_config(raw_data, base_dir=path.parent)
    return ExperimentConfig.from_dict(resolved)


def save_experiment_config(config: ExperimentConfig, filepath: str | Path) -> None:
    path = Path(filepath)
    if not path.parent.exists():
        raise ValueError(f"Couldn't save confige, dir not found: {path.parent}")

    data = config.to_dict()
    if path.suffix.lower() in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        raise ConfigurationError(
            f"Unsupported config format {path.suffix!r}; use .json, .yaml or .yml"
        )
