import fire
from cli import CLI


def main():
    fire.Fire(CLI)


try:
    main()
except BaseException as e:
    print(e)
