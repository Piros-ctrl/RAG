# from json import dumps
from chunking import chunking
from pick_chunk import Indexer


def main():
    path = "src/vllm-0.10.1"

    cls = Indexer()
    cls.build(1500, path)
    # cls = chunking()
    # output = cls.chunks(path, 1500)
    # print(output)
    # with open("output.json", "w")as file:
    #     file.write(dumps(output, indent=4))
    # print(output)


main()
