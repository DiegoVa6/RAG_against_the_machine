import fire

from .cli import RagCLI

#    llm = LLM(MODEL_PATH)
#    response = llm.generate_response(messages)
#    print(response)
#    recall = Recall(bmindex)
#    dataset = recall.read_file(
#        "data/datasets/AnsweredQuestions/dataset_docs_public.json")
#    recall.check_retrieval(dataset, 5)
#
#     resultados = bmindex.search("how to install vllm", k=5)
#     for chunk, nota in resultados:
#         print(nota, chunk.file_path, chunk.start_index, chunk.end_index)


if __name__ == "__main__":
    fire.Fire(RagCLI)
