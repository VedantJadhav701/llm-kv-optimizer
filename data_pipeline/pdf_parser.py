import fitz  # PyMuPDF
import re
import yaml
from pathlib import Path
from typing import List, Dict

def load_config(config_path: str = "configs/pipeline.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class PDFParser:
    """Utility to parse PDFs and clean research paper text."""
    
    def __init__(self, config: dict):
        self.config = config["data"]
        self.papers_dir = Path(self.config["papers_dir"])

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans text from noise, references, and normalized whitespace."""
        # Remove references section (common pattern)
        text = re.split(r'\nReferences\n|\nREFERENCES\n|\nBibliography\n', text, maxsplit=1)[0]
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove weird characters/noise typical in PDF extraction
        text = re.sub(r'\[\d+\]', '', text)  # [12] style citations
        text = re.sub(r'\(.*?\d{4}.*?\)', '', text) # (Author, 2020) style citations
        
        return text.strip()

    def parse_all(self) -> List[Dict[str, str]]:
        """Parses all PDFs in the papers directory."""
        parsed_data = []
        pdf_files = list(self.papers_dir.glob("*.pdf"))
        
        for pdf_path in pdf_files:
            print(f"Parsing: {pdf_path.name}")
            try:
                doc = fitz.open(pdf_path)
                full_text = ""
                for page in doc:
                    full_text += page.get_text()
                
                cleaned_text = self.clean_text(full_text)
                parsed_data.append({
                    "source": pdf_path.name,
                    "text": cleaned_text
                })
                doc.close()
            except Exception as e:
                print(f"Failed to parse {pdf_path.name}: {e}")
                
        return parsed_data

if __name__ == "__main__":
    cfg = load_config()
    parser = PDFParser(cfg)
    data = parser.parse_all()
    print(f"Total papers parsed: {len(data)}")
