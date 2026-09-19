from json import dumps
from chunking import chunking


def main():
    path = "src/vllm-0.10.1"

    cls = chunking()
    output = cls.chunks(1500, path)
    with open("output.json", "w")as file:
        file.write(dumps(output, indent=4))
    # print(output)

main()