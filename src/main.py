# from json import dumps
from pick_chunk import Indexer
from search import Retriever


def main():
    path = "src/vllm-0.10.1"

    cls = Indexer()
    cls.build(1500, path)
    instance = Retriever()
    outputs = instance.output("How do I install vLLM?")
    for output in outputs:
        print(output)
    # cls = chunking()
    # output = cls.chunks(path, 1500)
    # print(output)
    # with open("output.json", "w")as file:
    #     file.write(dumps(output, indent=4))
    # print(output)


main()
