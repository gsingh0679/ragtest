"""Document loading and text extraction"""

from pathlib import Path
from datetime import datetime
import pypdf
from src.models import Document


class DocumentLoader:
    """Load documents from various file formats (PDF, TXT, Markdown)"""

    SUPPORTED_FORMATS = {".pdf", ".txt", ".md", ".markdown"}

    @staticmethod
    def load_pdf(file_path: Path) -> str:
        """
        Extract text from PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text from all pages

        Raises:
            ValueError: If PDF is corrupted or cannot be read
        """
        try:
            pdf_reader = pypdf.PdfReader(file_path)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
            return text
        except Exception as e:
            raise ValueError(f"Failed to read PDF {file_path}: {str(e)}")

    @staticmethod
    def load_text(file_path: Path) -> str:
        """
        Extract text from plain text file.

        Args:
            file_path: Path to TXT file

        Returns:
            File content as string

        Raises:
            ValueError: If file cannot be decoded
        """
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return file_path.read_text(encoding="latin-1")
            except Exception as e:
                raise ValueError(
                    f"Failed to decode text file {file_path}: {str(e)}"
                )

    @staticmethod
    def load_markdown(file_path: Path) -> str:
        """
        Extract text from Markdown file.

        Args:
            file_path: Path to MD file

        Returns:
            File content as string (preserves markdown formatting)

        Raises:
            ValueError: If file cannot be read
        """
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Failed to read Markdown file {file_path}: {str(e)}")

    def load(self, file_path: str | Path) -> Document:
        """
        Load a single file and return Document object.

        Supports: PDF, TXT, MD/Markdown files

        Args:
            file_path: Path to file (string or Path object)

        Returns:
            Document object with content and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format unsupported or file cannot be read
        """
        file_path = Path(file_path)

        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Validate format is supported
        file_ext = file_path.suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {file_ext}. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )

        # Route to appropriate parser
        if file_ext == ".pdf":
            content = self.load_pdf(file_path)
            file_type = "pdf"
        elif file_ext == ".txt":
            content = self.load_text(file_path)
            file_type = "txt"
        elif file_ext in {".md", ".markdown"}:
            content = self.load_markdown(file_path)
            file_type = "md"

        # Capture file metadata
        stat = file_path.stat()

        # Return structured Document
        return Document(
            content=content,
            source=file_path.name,
            file_path=file_path,
            file_type=file_type,
            size_bytes=stat.st_size,
            loaded_at=datetime.now(),
        )

    def load_directory(self, dir_path: str | Path) -> list[Document]:
        """
        Load all supported files from a directory recursively.

        Args:
            dir_path: Path to directory

        Returns:
            List of Document objects

        Raises:
            NotADirectoryError: If path is not a directory
        """
        dir_path = Path(dir_path)

        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        documents = []
        for file_path in sorted(dir_path.rglob("*")):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.SUPPORTED_FORMATS
            ):
                try:
                    doc = self.load(file_path)
                    documents.append(doc)
                    print(f"✓ Loaded: {file_path.name} ({doc.file_type})")
                except Exception as e:
                    print(f"✗ Failed: {file_path.name} - {str(e)}")

        return documents
