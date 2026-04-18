import json
import yaml
from pathlib import Path
from typing import List, Dict

def load_config(config_path: str = "configs/pipeline.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class DatasetGenerator:
    """Processes cleaned text into instruction-format chunks for training."""
    
    def __init__(self, config: dict):
        self.config = config["data"]
        self.chunk_size = self.config["chunk_size"]
        self.chunk_overlap = self.config["chunk_overlap"]

    def chunk_text(self, text: str) -> List[str]:
        """Splits text into overlapping chunks of token-like words."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            if len(chunk) > 100: # Filter out tiny debris
                chunks.append(chunk)
        return chunks

    def generate_instruction_pair(self, chunk: str, paper_name: str) -> Dict[str, str]:
        """Generates a rule-based instruction pair for a chunk."""
        # Simple heuristic-based generation
        # In a real production system, this could be a call to a smaller model
        return {
            "instruction": "Summarize the key findings or methodology described in this section of the research paper.",
            "input": f"Source: {paper_name}\n\nContent: {chunk}",
            "output": f"The provided text from '{paper_name}' discusses various technical aspects. Based on the excerpt: {chunk[:200]}..." # Simple placeholder-ish logic for now
        }

    def process(self, parsed_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        dataset = []
        for item in parsed_data:
            paper_name = item["source"]
            text = item["text"]
            chunks = self.chunk_text(text)
            
            for chunk in chunks:
                pair = self.generate_instruction_pair(chunk, paper_name)
                dataset.append(pair)
                
        return dataset

    def save_dataset(self, dataset: List[Dict[str, str]]):
        output_path = Path(self.config["dataset_path"])
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4)
        print(f"Saved {len(dataset)} examples to {output_path}")

if __name__ == "__main__":
    import sys
    import os
    # Add project root to sys.path for absolute imports
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
    from data_pipeline.pdf_parser import PDFParser
    cfg = load_config()
    
    # 1. Parse
    parser = PDFParser(cfg)
    parsed_data = parser.parse_all()
    
    # 2. Generate Dataset
    generator = DatasetGenerator(cfg)
    dataset = generator.process(parsed_data)
    
    # 3. Save
    generator.save_dataset(dataset)
