from rich.console import Console

MODE_ASK = "ASK"
MODE_SOLVE = "SOLVE"

_COMMANDS = [
    "/clear",
    "/mode ask",
    "/mode solve",
    "/apikey",
    "/help",
    "/load",
    "/reasoning-effort",
]

_OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"

console = Console()
