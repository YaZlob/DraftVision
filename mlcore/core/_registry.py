import copy
import logging
from typing import Any
from dataclasses import dataclass
from collections.abc import Callable, Iterator, Mapping, Sequence

from ._config import ComponentConfig
from ._contracts import ComponentType, Task, _get_task
from ._errors import (
    ComponentLookupError,
    CompatibilityError,
    RegistrationError,
    BuildComponentError,
    ConfigurationError,
)

Factory = Callable[..., Any]

_TASK_FACING_COMPONENTS = frozenset(
    {
        ComponentType.MODEL,
        ComponentType.LOSS,
        ComponentType.METRIC,
        ComponentType.DATASET,
        ComponentType.COLLATE_FN,
    }
)
_TASK_INDEPENDENT_COMPONENTS = frozenset(
    {
        ComponentType.TRANSFORMS,
        ComponentType.OPTIMIZER,
        ComponentType.SCHEDULER,
        ComponentType.DATALOADER,
        ComponentType.EMA,
    }
)
logger = logging.getLogger()


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """Metadata and factory stored for a registered component."""

    name: str
    ctype: ComponentType
    factory: Factory
    supported_tasks: frozenset[Task] = frozenset()
    provider: str = "native"
    description: str | None = None

    def supports(self, task: Task | None) -> bool:
        return task is None or not self.supported_tasks or task in self.supported_tasks


class Registry:
    def __init__(self) -> None:
        self._components: dict[str, ComponentSpec] = {}

    def register(
        self,
        name: str,
        ctype: ComponentType | str,
        supported_tasks: Task | str | Sequence[Task] | None = None,
        provider: str = "native",
        description: str | None = None,
    ) -> Callable[[Factory], Factory]:
        """Register a factory as a decorator."""
        if not isinstance(name, str) or not name.strip():
            raise RegistrationError("component name must be a non-empty string")
        if not isinstance(provider, str) or not provider.strip():
            raise RegistrationError("component provider must be a non-empty string")

        try:
            ctype: ComponentType = ComponentType(ctype)
            supported_tasks: frozenset[Task] = _get_task(supported_tasks)
        except ValueError as err:
            raise RegistrationError(str(err)) from err

        if ctype in _TASK_FACING_COMPONENTS:
            if supported_tasks is None:
                raise RegistrationError(
                    f"{ctype.value} component {name!r} " "must declare supported_tasks"
                )
        if ctype in _TASK_INDEPENDENT_COMPONENTS and supported_tasks is not None:
            raise RegistrationError(
                f"{ctype.value} component {name!r} " "cannot declare supported_tasks"
            )

        def decorator(factory: Factory) -> Factory:
            if not callable(factory):
                raise RegistrationError(f"Factory for {name!r} must be callable")
            if name in self._components:
                component = self._components[name]
                raise RegistrationError(
                    f"Component {name!r} is already registered as\n {component}"
                )

            self._components[name] = ComponentSpec(
                name=name,
                ctype=ctype,
                factory=factory,
                supported_tasks=supported_tasks,
                provider=provider,
                description=description,
            )
            return factory

        return decorator

    def _resolve(
        self,
        name: str | None,
        ctype: ComponentType | str | None = None,
        task: Task | str | None = None,
    ) -> ComponentSpec:

        if name is None:
            if ctype is None or task is None:
                raise BuildComponentError(
                    "component name or (ctype, task) must be provided for building"
                )
            try:
                task, ctype = _get_task(task), ComponentType(ctype)
            except ValueError:
                raise BuildComponentError(
                    f"Provided incorrect `task` <{task}> or `ctype` <{ctype}>"
                )
            available: tuple[ComponentSpec] = self.available(ctype, task)
            if not available:
                raise BuildComponentError(
                    f"Component type: {ctype.value} for {task.value} not found"
                )
            elif len(available) > 1:
                msg = f"Found {len(available)} components for provided prms: {task.value}, {ctype.value}"
                msg += ", ".join(available)
                raise BuildComponentError(msg)

            return available[0]

        if not isinstance(name, str) or not name.strip():
            raise BuildComponentError("component name must be a non-empty string")

        try:
            component = self._components[name]
        except KeyError as err:
            available = ", ".join(self._components) or "<none>"
            raise ComponentLookupError(
                f"Unknown component {name!r}. Available components: {available}"
            ) from err

        if ctype and component.ctype != ctype:
            raise CompatibilityError(
                f"Component {name!r} is registered as {component.ctype.value}, "
                f"expected {ctype.value}"
            )
        if task and not component.supports(task):
            raise CompatibilityError(
                f"Component {name!r} does not support task {task.value}"
            )

        return component

    def resolve_spec(self, name, ctype=None, task=None) -> ComponentSpec:
        return self._resolve(name, ctype, task)

    def build(
        self,
        config: ComponentConfig,
        task: Task | str | None = None,
        config_path: str | None = None,
        **dependencies: dict,
    ) -> Any:
        """Validate and construct one component from config parameters."""

        component = self._resolve(config.type, config.ctype, task)
        config_path = config_path or config.ctype.value
        logger.debug(f"Build {component.ctype.value}, path: {config_path}")

        # Deepcopy prevents chandes for runtime building components
        # So later `ExperimentConfig` could be serialized
        component_prms = copy.deepcopy(config.params)
        if isinstance(config.components, Mapping):
            built_components = {
                role: self.build(child, config_path=f"{config_path}.components.{role}")
                for role, child in config.components.items()
            }
            duplicate_components = set(component_prms).intersection(built_components)
            if duplicate_components:
                keys = ", ".join(sorted(duplicate_components))
                raise ConfigurationError(
                    f"{config_path} params conflict with configured components: {keys}"
                )
            component_prms.update(built_components)

        elif config.components:
            if "components" in component_prms:
                raise ConfigurationError(
                    f"{config_path}.params.components conflicts with configured components"
                )
            component_prms["components"] = [
                self.build(child, config_path=f"{config_path}.components[{index}]")
                for index, child in enumerate(config.components)
            ]

        duplicate = set(component_prms).intersection(dependencies)
        if duplicate:
            keys = ", ".join(sorted(duplicate))
            raise ConfigurationError(
                f"{config_path} dependencies override configured params: {keys}"
            )
        component_prms.update(dependencies)
        try:
            return component.factory(**component_prms)
        except Exception as err:
            raise BuildComponentError(
                f"Couldn't build `{config.type}` as {config.ctype.value} "
                f"with params: {component_prms}"
            ) from err

    def available(
        self,
        ctype: ComponentType | str | None = None,
        task: Task | str | None = None,
    ) -> tuple[ComponentSpec, ...]:

        task = Task(task) if task else None
        ctype = ComponentType(ctype) if ctype else None

        return tuple(
            component
            for component in sorted(
                self._components.values(), key=lambda item: item.name
            )
            if (ctype and component.ctype == ctype) and component.supports(task)
        )

    def __contains__(self, name: object) -> bool:
        return name in self._components

    def __iter__(self) -> Iterator[ComponentSpec]:
        return iter(self.available())

    def __len__(self) -> int:
        return len(self._components)


COMPONENTS = Registry()
