"""What a recording fake sandbox saw a ``sudo bash -c`` exec run."""


def script_run(args: tuple[str, ...]) -> str:
    """The script of ``("sudo", "bash", "-c", script, name, *params)``, each ``"$n"`` replaced by its param."""
    script = args[3]
    for n, param in enumerate(args[5:], 1):
        script = script.replace(f'"${n}"', param)
    return script
