from dataclasses import dataclass
import re


@dataclass
class Chunk:
	text: str
	start_index: int
	end_index: int


class Chunker:
	def __init__(self, chunk_size=500, overlap=50):
		if overlap >= chunk_size:
			raise ValueError(
				f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})"
			)

		self.chunk_size = chunk_size
		self.overlap = overlap

	def chunk_text(self, text):
		chunks = []
		start = 0
		end = 0  # end of last chunk
		while end < len(text):
			end = min(start + self.chunk_size, len(text))
			chunk_text = text[start:end]
			chunks.append(Chunk(text=chunk_text, start_index=start, end_index=end))
			start += self.chunk_size - self.overlap
		return chunks

	def tokenize(self, line):
		line = line.lower()
		tokens = re.findall(r"\w+", line)
		return tokens


with open("data/raw/vllm-0.10.1/README.md", "r", encoding="utf-8") as f:
	readme_content = f.read()
	chunker = Chunker(chunk_size=1000, overlap=200)
	chunks = chunker.chunk_text(readme_content)
	for c in chunks:
		if readme_content[c.start_index:c.end_index] != c.text:
			print("NO COINCIDE:", c.start_index, c.end_index)
	for c in chunks:
		print(c.start_index, c.end_index, len(c.text))
	print(chunker.tokenize("vLLM supports LoRA adapters. Use enable_lora=True."))
	print(chunker.tokenize(chunks[3].text))
