import fire
from cli import CLI


def main():
    fire.Fire(CLI)

main()
# try:
#     main()
# except BaseException as e:
#     print(e)
