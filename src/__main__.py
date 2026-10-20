# from json import dumps
import fire
from search import Retriever
from cli import CLI
from pick_chunk import Indexer


def main():
    fire.Fire(CLI)
    # path = "data/raw/vllm-0.10.1"

    # cls = Indexer()
    # cls.build(path ,1500)
                # instance = Retriever()
                # outputs = instance.output("How do I install vLLM?")
                # for output in outputs:
                #     print(output)
    # cls = chunking()
    # output = cls.chunks(path, 1500)
    # print(output)
    # with open("output.json", "w")as file:
    #     file.write(dumps(output, indent=4))
    # print(output)


main()
