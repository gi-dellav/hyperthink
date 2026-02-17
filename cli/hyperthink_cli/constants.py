from rich.console import Console

MODE_ASK = "ASK"
MODE_SOLVE = "SOLVE"
MODE_PLAN = "PLAN"

_COMMANDS = [
    "/clear",
    "/mode ask",
    "/mode solve",
    "/mode plan",
    "/apikey",
    "/help",
    "/load",
    "/reasoning-effort",
    "/mcp",
]

_OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"

console = Console()
