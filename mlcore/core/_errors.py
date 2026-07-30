class CoreError(Exception):
    """Base error for mlcore infrastructure failures."""


class RegistrationError(CoreError):
    """Raised when a component cannot be registered."""


class ComponentLookupError(CoreError):
    """Raised when a requested component is not registered."""


class CompatibilityError(CoreError):
    """Raised when a component does not satisfy an experiment contract."""


class ConfigurationError(CoreError):
    """Raised when an experiment configuration is malformed."""


class BuildComponentError(CoreError):
    """Raised when component buildings fails"""
