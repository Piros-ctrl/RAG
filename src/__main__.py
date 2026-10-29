import fire
from cli import CLI


def main() -> None:
    """Run the CLI class through Fire."""
    fire.Fire(CLI)


try:
    main()
except BaseException as e:
    print(e)
