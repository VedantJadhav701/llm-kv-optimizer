import arxiv
import os
import yaml
from pathlib import Path
from typing import List

def load_config(config_path: str = "configs/pipeline.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class ArxivDownloader:
    """Utility class to download research papers from arXiv."""
    
    def __init__(self, config: dict):
        self.config = config["data"]
        self.papers_dir = Path(self.config["papers_dir"])
        self.papers_dir.mkdir(exist_ok=True)

    def search_and_download(self) -> List[str]:
        """Searches for papers and downloads the PDFs."""
        print(f"Searching for: {self.config['arxiv_search_query']}")
        
        search = arxiv.Search(
            query=self.config["arxiv_search_query"],
            max_results=self.config["max_papers"],
            sort_by=arxiv.SortCriterion.Relevance
        )

        client = arxiv.Client()
        downloaded_files = []
        for result in client.results(search):
            filename = f"{result.entry_id.split('/')[-1]}.pdf"
            filepath = self.papers_dir / filename
            
            if filepath.exists():
                print(f"Skipping {filename}, already exists.")
                downloaded_files.append(str(filepath))
                continue
                
            print(f"Downloading: {result.title}")
            try:
                result.download_pdf(dirpath=str(self.papers_dir), filename=filename)
                downloaded_files.append(str(filepath))
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                
        return downloaded_files

if __name__ == "__main__":
    cfg = load_config()
    downloader = ArxivDownloader(cfg)
    downloader.search_and_download()
