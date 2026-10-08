"""Run the consolidated tools in their own directories and environments."""

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "CONSOLIDATION.json").read_text())
    tasks = manifest["tasks"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("task", nargs="?", choices=sorted(tasks))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.list or args.task is None:
        for name, task in tasks.items():
            print(f"{name}: {task['description']}")
        return
    task = tasks[args.task]
    cwd = (root / task["directory"]).resolve()
    assert cwd.is_relative_to(root) and cwd.is_dir(), "Invalid task directory"
    for relative in task["files"]:
        assert (cwd / relative).is_file(), f"Missing entry point: {relative}"
    command = ["uv", "run", "--no-project", "--isolated", "--python", task.get("python", "3.12")]
    for requirement in task["dependencies"]:
        if os.uname().sysname != "Linux":
            requirement = requirement.replace("jax[cuda12]", "jax")
        command.extend(["--with", requirement])
    command.extend(task["command"])
    extra = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    command.extend(extra)
    if args.dry_run:
        print(f"Directory: {cwd}")
        print(shlex.join(command))
        return
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(cwd / "src"), str(cwd)])
    environment.pop("VIRTUAL_ENV", None)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


if __name__ == "__main__":
    main()
