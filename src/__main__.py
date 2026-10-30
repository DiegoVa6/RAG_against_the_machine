import fire

from .cli import RagCLI

#    recall = Recall(bmindex)
#    dataset = recall.read_file(
#        "data/datasets/AnsweredQuestions/dataset_docs_public.json")
#    recall.check_retrieval(dataset, 5)

if __name__ == "__main__":
    fire.Fire(RagCLI)
