# PDF to Markdown Pipeline

## Project Overview

We've built a sophisticated, high-fidelity PDF to Markdown conversion pipeline that preserves mathematical formulas, schemas, diagrams, and complex layouts using LangChain's ChatOllama integration.

<span style="color: red;">HYBRID PARSER NOT WORKING PROPERLY</span>

## Architecture

### Core Components

1. **Utils & Configuration** (`utils.py`)
   - Content type detection (text, images, tables, formulas)
   - Image optimization for vision models
   - Text formatting preservation utilities
   - Pipeline configuration management

2. **PDF Analyzer** (`pdf_analyzer.py`)
   - Intelligent page analysis
   - Strategy determination (text-only, vision-only, hybrid)
   - Layout complexity assessment
   - Content type detection

3. **Text Extractor** (`text_extractor.py`)
   - Direct PDF text extraction with formatting
   - Mathematical formula detection
   - Structure preservation (headers, lists, emphasis)
   - Font and style analysis

4. **Vision Processor** (`vision_processor.py`)
   - LangChain ChatOllama integration
   - Specialized prompts for tables, diagrams, formulas
   - Multi-modal content processing
   - Configurable model and base_url

5. **Content Integrator** (`content_integrator.py`)
   - Smart merging of text and vision extractions
   - Conflict resolution between overlapping content
   - Document flow preservation
   - Quality-based content selection

6. **Markdown Generator** (`markdown_generator.py`)
   - High-fidelity markdown formatting
   - LaTeX formula preservation
   - Table structure maintenance
   - Image embedding (base64 or file references)

7. **Main Pipeline** (`main_pipeline.py`)
   - Orchestrates entire conversion process
   - Page-by-page processing
   - Error handling and recovery
   - Result serialization

## Key Features

- **Adaptive Strategy**: Automatically chooses optimal processing method per page
- **Content-Aware**: Detects tables, formulas, diagrams, and complex layouts
- **Mathematical Formulas**: Tries to preserve as LaTeX notation
- **Custom Base URL**: Remote Ollama servers supported
- **Processing Options**: Text-only, vision-only, or hybrid modes
- **Output Formats**: Individual pages or combined documents

## Usage Examples

### Simple Conversion
```python
from pdf_markdown_pipeline import convert_pdf_to_markdown

result = convert_pdf_to_markdown(
    pdf_path="document.pdf",
    ollama_model="llama3.2-vision:11b",
    ollama_base_url="http://localhost:11434"
)
```

### Advanced Configuration
```python
from pdf_markdown_pipeline import PDFToMarkdownPipeline, PipelineConfig

config = PipelineConfig()
config.dpi = 400
config.vision_model_temp = 0.1
config.preserve_formatting = True

pipeline = PDFToMarkdownPipeline("llama3.2-vision:11b", "http://localhost:11434", config)
result = pipeline.convert_pdf("document.pdf")
```

### Content-Specific Extraction
```python
# Extract only tables and formulas
page_image = Utils.extract_page_image(page, 300)
table_result = vision_processor.extract_table_data(page_image)
formula_result = vision_processor.extract_formulas(page_image)
```

## Configuration Options

### PipelineConfig
- `dpi`: Image resolution (default: 300)
- `vision_model_temp`: Temperature for vision model (default: 0.1)
- `text_extraction_priority`: Prefer text extraction (default: True)
- `preserve_formatting`: Maintain original formatting (default: True)
- `image_embed_mode`: "base64" or "file" (default: "base64")

### Processing Strategies
- **TEXT_ONLY**: Pure text extraction for simple documents
- **VISION_ONLY**: Vision model for scanned/complex documents
- **HYBRID**: Combines text + vision for optimal results

## Dependencies

- `PyMuPDF` (fitz): PDF processing
- `langchain-ollama`: LangChain Ollama integration
- `PIL` (Pillow): Image processing
- `pydantic`: Data validation

## Getting Started

1. **Install Ollama** and pull a vision model:
   ```bash
   ollama pull llama3.2-vision:11b
   ```

2. **Install Python dependencies**:
   ```bash
   pip install PyMuPDF langchain-ollama pillow pydantic
   ```

3. **Run the pipeline**:
   ```python
   from pdf_markdown_pipeline import convert_pdf_to_markdown
   
   result = convert_pdf_to_markdown("your_document.pdf")
   ```