import logging
from typing import Literal
from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Mapping, Callable
from ._template import SolverTemplate
from ._errors import RegistrationError, BuildCallbackError, ConfigurationError
from ._config import CallbackConfig

logger = logging.getLogger()
CALLBACKHOOKS = Literal[
    "on_train_start",
    "on_train_end",
    "on_epoch_start",
    "on_epoch_end",
    # Will be implemented later
    # "on_train_step_start",
    # "on_train_step_end",
    # "on_val_start",
    # "on_val_end",
]
_CALLBACKHOOKS: frozenset[str] = frozenset(CALLBACKHOOKS.__args__)


class Callback:
    is_triggered = False

    def __init__(self, solver: SolverTemplate):
        assert isinstance(solver, SolverTemplate)
        self._solver = solver

    # Not requires additional metadata, everything could be extracted from solver
    def on_train_start(self): ...
    def on_train_end(self): ...
    def on_epoch_start(self): ...
    def on_epoch_end(self): ...


@dataclass(frozen=True, slots=True)
class CallBackSchema:
    name: str
    when: CALLBACKHOOKS
    callback: Callback  # callback factory
    persistent: bool

    def __post_init__(self):
        if self.when not in _CALLBACKHOOKS:
            raise RegistrationError(f"unsupported hook {self.when}")


@dataclass(frozen=True)
class Hook:
    name: str
    persistent: bool
    hook: Callback

    def __repr__(self):
        return f"Name: {self.name}-persitent: {self.persistent}-{self.hook.__class__.__name__}"


class CallBackRegistry:
    def __init__(self):

        self._callbacks: dict[CALLBACKHOOKS, list[Hook]] = defaultdict(list)
        self.__callback_factories: dict[str, CallBackSchema] = {}

    def register(self, name: str, when: CALLBACKHOOKS, persistent: bool = False):
        if not isinstance(name, str) or not name.strip():
            raise RegistrationError("callback name must be non-empty string")

        if when not in _CALLBACKHOOKS:
            raise RegistrationError(
                f"Unknown callback hook {when!r}. "
                f"Expected one of: {', '.join(sorted(_CALLBACKHOOKS))}"
            )

        def decorator(callback: Callable):
            if not callable(callback):
                raise RegistrationError(f"Callback {name} must be callable")
            if name in self.__callback_factories:
                raise RegistrationError(f"Callback {name} is already registered")

            if not issubclass(callback, Callback):
                raise ConfigurationError("callback must inherit from `CallBack`")

            self.__callback_factories[name] = CallBackSchema(
                name=name, callback=callback, when=when, persistent=persistent
            )
            return callback

        return decorator

    def build(
        self, config: CallbackConfig, config_path: str, **dependencies: Mapping
    ) -> None:

        name, params = config.type, config.params
        try:
            callbackschema = self.__callback_factories.get(name)
        except KeyError:
            available = ", ".join(self.__callback_factories) or "<none>"
            raise KeyError(
                f"Callback with name: {name}. Available callbacks: {available}"
            )

        merged = dict(params or {})
        duplicate = set(params).intersection(dependencies)
        if duplicate:
            keys = ", ".join(sorted(duplicate))
            raise ConfigurationError(f"Dependencies override configured params: {keys}")

        merged.update(dependencies)

        try:
            hook = Hook(
                name=name,
                persistent=callbackschema.persistent,
                hook=callbackschema.callback(**merged),
            )
            self._callbacks[callbackschema.when].append(hook)

        except Exception as err:
            raise BuildCallbackError(
                f"Couldn't build callback `{name}` with params: {merged}, path: {config_path}"
            ) from err

    def call_hooks(self, hookname: CALLBACKHOOKS):
        for hook in self._callbacks[hookname]:
            logger.debug(f"Call {hook.name} ...")
            getattr(hook.hook, hookname)()

        # Clear triggered callbacks
        self._callbacks[hookname] = [
            hook
            for hook in self._callbacks[hookname]
            if hook.persistent or not hook.hook.is_triggered
        ]

    def __repr__(self) -> str:
        msg = ""
        for ht, hooks in self._callbacks.items():
            msg = f"Hook type: {ht}, available hooks:\n"
            for hook in hooks:
                msg += f"{hook}\n"
        return msg
