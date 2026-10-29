import fire
from src.cli import CLI


def main() -> None:
    """Run the CLI class through Fire."""
    fire.Fire(CLI)


main()
# try:
#     main()
# except BaseException as e:
#     print(e)
