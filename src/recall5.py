from .bm25 import BM25Index
import json


class Recall:
    def __init__(self, index: BM25Index):
        self.bm25index = index

    def read_file(self, file_path: str) -> str:
        """Return the full content of a file as text."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            return json.loads(text)

    def ask_question(self, question: str, k = 5):
        return self.bm25index.search(question, k)

    def overlap(self, start1: int, end1: int, start2: int, end2: int) -> int:
        s = max(start1, start2)
        e = min(end1, end2)
        return max(0, (e - s))

    def iou(self, start1: int, end1: int, start2: int, end2: int) -> float:
        iter = self.overlap(start1, end1, start2, end2)
        largo1 = end1 - start1
        largo2 = end2 - start2
        union = largo1 + largo2 - iter
        if union == 0:
            return 0.0
        return iter / union

    def check_retrieval(self, text_json, k=5) -> float:
        questions = text_json["rag_questions"]
        aciertos = 0

        for question in questions:
            result = self.ask_question(question["question"], k)
            source = question["sources"][0]

            acierto = False
            for chunk, nota in result:
                if chunk.file_path != source["file_path"]:
                    continue  # otro archivo: no puede acertar
                valor = self.iou(chunk.start_index, chunk.end_index,
                                 source["first_character_index"],
                                 source["last_character_index"])
                if valor > 0.05:
                    acierto = True
                    break  # con uno basta

            if acierto:
                aciertos += 1
            else:
                print("FALLO:", question["question"])

        recall = aciertos / len(questions)
        print(f"recall@5: {aciertos}/{len(questions)} = {recall:.2f}")
        return recall
