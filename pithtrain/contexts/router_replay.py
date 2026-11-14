"""Identity registry for built-in routing callbacks that cannot install observers."""

import weakref
from types import FunctionType

observer_free_replays = weakref.WeakKeyDictionary()


def register_observer_free_replay(callback):
    """Register a built-in callback while retaining its original code identity."""
    observer_free_replays[callback] = callback.__code__
    return callback


def is_observer_free_replay(callback):
    """Unknown, wrapped or code-replaced routing callbacks require ordinary execution."""
    return (
        type(callback) is FunctionType and observer_free_replays.get(callback) is callback.__code__
    )
